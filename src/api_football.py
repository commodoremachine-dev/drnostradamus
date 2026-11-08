"""API-Football (api-sports.io) üzerinden Süper Lig verisi çeker.

Anahtar: proje klasöründeki .env dosyasına  API_FOOTBALL_KEY=anahtarınız  yazın
(ya da aynı adla ortam değişkeni tanımlayın). Anahtar hiçbir yere yazdırılmaz.

Kullanım:
  python src/api_football.py --sezon 2026                # 2026-27 sezonu
  python src/api_football.py --sezon 2024 2025 2026 --max-istek 60
  python src/api_football.py --donustur                  # sadece önbellekten CSV üret

Ne indirir (sezon başına yaklaşık 18 istek):
  /fixtures?league=203&season=S      -> fikstür, skor, ilk yarı skoru   (1 istek)
  /fixtures?ids=a-b-...  (20'şer)    -> şut, isabetli şut, korner, kart,
                                        topla oynama, xG                (~16 istek)
  /injuries?league=203&season=S      -> sakat/cezalı oyuncular          (1 istek)

Biten maçların detayları bir kez indirilir ve data/raw/api/ altında saklanır;
tekrar çalıştırınca yalnızca yeni biten maçlar için istek atılır.
Çıktı: data/api_matches.csv  (parse_openfootball.py bunu otomatik birleştirir)
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from teams import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "api"
BASE = "https://v3.football.api-sports.io"
LEAGUE_ID = 203  # Süper Lig
FINISHED = {"FT", "AET", "PEN"}
DROP = {"CANC", "ABD", "AWD", "WO"}
STAT_MAP = {  # API istatistik adı -> (ev sütunu, deplasman sütunu)
    "Total Shots": ("hs", "as"),
    "Shots on Goal": ("hst", "ast"),
    "Corner Kicks": ("hc", "ac"),
    "Yellow Cards": ("hy", "ay"),
    "Red Cards": ("hr", "ar"),
    "expected_goals": ("hxg", "axg"),
    "Ball Possession": ("hposs", "aposs"),
    "Shots insidebox": ("hsib", "asib"),
    "Passes %": ("hpass", "apass"),
}
LIST_MAX_AGE_H = 6
# Yorgunluk hesabı için diğer turnuvalar (yalnızca tarihleri kullanılır)
OTHER_LEAGUES = {206: "Türkiye Kupası", 2: "Şampiyonlar Ligi", 3: "Avrupa Ligi", 848: "Konferans Ligi"}  # fikstür listesi bu kadar saatten eskiyse yenilenir


class ApiError(RuntimeError):
    pass


def load_key() -> str | None:
    key = os.environ.get("API_FOOTBALL_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("API_FOOTBALL_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key or None


class Client:
    def __init__(self, key: str | None, max_requests: int):
        self.key = key
        self.budget = max_requests
        self.used = 0
        self.remaining_day = None

    def get(self, endpoint: str, params: dict) -> list:
        if not self.key:
            raise ApiError("API anahtarı yok. .env dosyasına API_FOOTBALL_KEY=... ekleyin.")
        if self.used >= self.budget:
            raise ApiError(f"Bu çalıştırma için istek sınırına ulaşıldı ({self.budget}).")
        url = f"{BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"x-apisports-key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                rem = resp.headers.get("x-ratelimit-requests-remaining")
                per_min = resp.headers.get("X-RateLimit-Remaining")
        except urllib.error.HTTPError as e:
            raise ApiError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
        self.used += 1
        if rem is not None:
            self.remaining_day = int(rem)
        errs = body.get("errors")
        if errs:
            msg = "; ".join(f"{k}: {v}" for k, v in errs.items()) if isinstance(errs, dict) else str(errs)
            raise ApiError(msg)
        if per_min is not None and int(per_min) <= 1:
            time.sleep(60)  # dakika başı sınır
        return body.get("response", [])


# ------------------------------------------------------------------ indirme
def season_dir(season: int) -> Path:
    d = CACHE / str(season)
    (d / "fixtures").mkdir(parents=True, exist_ok=True)
    return d


def fetch_season(cli: Client, season: int, force_list=False):
    d = season_dir(season)
    lst = d / "fixtures_list.json"
    fresh = lst.exists() and (time.time() - lst.stat().st_mtime) < LIST_MAX_AGE_H * 3600
    if force_list or not fresh:
        data = cli.get("fixtures", {"league": LEAGUE_ID, "season": season})
        lst.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"  {season}: fikstür listesi indirildi ({len(data)} maç)")
    fixtures = json.loads(lst.read_text(encoding="utf-8"))

    need = [f["fixture"]["id"] for f in fixtures
            if f["fixture"]["status"]["short"] in FINISHED
            and not (d / "fixtures" / f"{f['fixture']['id']}.json").exists()]
    print(f"  {season}: istatistiği eksik biten maç: {len(need)}")
    for i in range(0, len(need), 20):
        batch = need[i:i + 20]
        data = cli.get("fixtures", {"ids": "-".join(map(str, batch))})
        for fx in data:
            fid = fx["fixture"]["id"]
            # Sadece gerekli kısmı sakla (oyuncu/olay verisi büyük)
            slim = {k: fx.get(k) for k in ("fixture", "league", "teams", "goals", "score", "statistics")}
            slim["players_slim"] = _slim_players(fx.get("players") or [])
            slim["events_slim"] = [
                {"team_id": (e.get("team") or {}).get("id"), "type": e.get("type"), "detail": e.get("detail")}
                for e in (fx.get("events") or [])
                if e.get("type") in ("Goal", "Card", "Var")]
            (d / "fixtures" / f"{fid}.json").write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")

    for lid in OTHER_LEAGUES:
        op = d / f"other_{lid}.json"
        if op.exists() and (time.time() - op.stat().st_mtime) < 24 * 3600:
            continue
        try:
            data = cli.get("fixtures", {"league": lid, "season": season})
        except ApiError as e:
            if "plan" in str(e).lower():
                print(f"  {season}: {OTHER_LEAGUES[lid]} planda yok, atlandı")
                continue
            raise
        slim = [{"date": f["fixture"]["date"], "status": f["fixture"]["status"]["short"],
                 "home": f["teams"]["home"]["name"], "away": f["teams"]["away"]["name"]} for f in data]
        op.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")

    inj = d / "injuries.json"
    inj_fresh = inj.exists() and (time.time() - inj.stat().st_mtime) < LIST_MAX_AGE_H * 3600
    if not inj_fresh:
        data = cli.get("injuries", {"league": LEAGUE_ID, "season": season})
        inj.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"  {season}: sakat/cezalı kaydı: {len(data)}")


def _num(x):
    try:
        return float(x) if x is not None and x != "" else 0.0
    except (TypeError, ValueError):
        return 0.0


def _slim_players(players: list) -> dict:
    """/fixtures?ids= yanıtındaki oyuncu istatistiklerini küçük bir yapıya indirger."""
    out = {}
    for team in players:
        tid = str(team["team"]["id"])
        rows = []
        for p in team.get("players", []):
            s = (p.get("statistics") or [{}])[0]
            g = s.get("games") or {}
            mins = _num(g.get("minutes"))
            if mins <= 0:
                continue
            rows.append(dict(
                id=p["player"]["id"], name=p["player"].get("name"), min=mins,
                rating=_num(g.get("rating")) or None,
                goals=_num((s.get("goals") or {}).get("total")),
                assists=_num((s.get("goals") or {}).get("assists")),
                key=_num((s.get("passes") or {}).get("key")),
                tkl=_num((s.get("tackles") or {}).get("total")) + _num((s.get("tackles") or {}).get("interceptions")),
                duels=_num((s.get("duels") or {}).get("total")), duels_won=_num((s.get("duels") or {}).get("won")),
                drb=_num((s.get("dribbles") or {}).get("attempts")), drb_ok=_num((s.get("dribbles") or {}).get("success")),
            ))
        out[tid] = rows
    return out


def _team_player_aggs(rows: list) -> dict:
    if not rows:
        return {}
    mins = sum(r["min"] for r in rows)
    rated = [(r["rating"], r["min"]) for r in rows if r.get("rating")]
    duels = sum(r["duels"] for r in rows)
    drb = sum(r["drb"] for r in rows)
    return {
        "duel": 100 * sum(r["duels_won"] for r in rows) / duels if duels else None,
        "drb": 100 * sum(r["drb_ok"] for r in rows) / drb if drb else None,
        "kp": sum(r["key"] for r in rows),
        "tkl": sum(r["tkl"] for r in rows),
        "rating": sum(a * b for a, b in rated) / sum(b for _, b in rated) if rated else None,
        "_mins": mins,
    }


# ------------------------------------------------------------------ dönüştürme
def _stat_value(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip().rstrip("%")
        try:
            return float(v)
        except ValueError:
            return None
    return float(v)


def convert() -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in CACHE.glob("*") if p.is_dir() and p.name.isdigit()):
        season = int(d.name)
        lst = d / "fixtures_list.json"
        if not lst.exists():
            continue
        fixtures = sorted(json.loads(lst.read_text(encoding="utf-8")),
                          key=lambda f: f["fixture"]["date"])
        missing = Counter()
        missing_ids = defaultdict(list)  # (fixture, takım) -> eksik oyuncu id'leri
        inj = d / "injuries.json"
        if inj.exists():
            for it in json.loads(inj.read_text(encoding="utf-8")):
                if it.get("player", {}).get("type") in ("Missing Fixture", None):
                    k = (it["fixture"]["id"], it["team"]["id"])
                    missing[k] += 1
                    if it.get("player", {}).get("id") is not None:
                        missing_ids[k].append(it["player"]["id"])
        # Sezon içi oyuncu katkıları (yalnızca o maçtan önceki maçlar)
        p_min = defaultdict(lambda: defaultdict(float))   # takım -> oyuncu -> dakika
        p_ga = defaultdict(lambda: defaultdict(float))    # takım -> oyuncu -> gol+asist
        for f in fixtures:
            fx = f["fixture"]
            status = fx["status"]["short"]
            if status in DROP:
                continue
            detail_p = d / "fixtures" / f"{fx['id']}.json"
            detail = json.loads(detail_p.read_text(encoding="utf-8")) if detail_p.exists() else f
            # Türkiye saati (UTC+3) ile tarih
            dt = pd.Timestamp(fx["date"]).tz_convert("UTC") + pd.Timedelta(hours=3)
            played = status in FINISHED
            ht = (detail.get("score") or {}).get("halftime") or {}
            th, ta = detail["teams"]["home"], detail["teams"]["away"]
            ref = (detail.get("fixture") or fx).get("referee") or fx.get("referee")
            row = dict(
                season=f"{season}-{(season + 1) % 100:02d}", date=dt.date(),
                round=_round_no(f["league"].get("round")),
                home=normalize(th["name"]), away=normalize(ta["name"]),
                hg=detail["goals"]["home"] if played else None,
                ag=detail["goals"]["away"] if played else None,
                hthg=ht.get("home") if played else None, htag=ht.get("away") if played else None,
                played=played, fixture_id=fx["id"],
                referee=ref.split(",")[0].strip() if ref else None,
                h_missing=missing[(fx["id"], th["id"])] if inj.exists() else None,
                a_missing=missing[(fx["id"], ta["id"])] if inj.exists() else None,
            )
            # Kilit oyuncu eksikliği: eksiklerin sezon dakika payı (1.0 = bir tam zamanlı
            # ilk 11 oyuncusu) ve takımın gol+asistindeki payı
            for pre, team in (("h", th), ("a", ta)):
                tid = team["id"]
                tot_min = sum(p_min[tid].values())
                tot_ga = sum(p_ga[tid].values())
                ids = missing_ids.get((fx["id"], tid), [])
                if inj.exists() and tot_min >= 90 * 11 * 3:
                    row[f"{pre}_missing_min_share"] = 11 * sum(p_min[tid].get(i, 0) for i in ids) / tot_min
                    row[f"{pre}_missing_ga_share"] = (sum(p_ga[tid].get(i, 0) for i in ids) / tot_ga
                                                     if tot_ga else 0.0)
            for team_stats in detail.get("statistics") or []:
                side = 0 if team_stats["team"]["id"] == th["id"] else 1
                for s in team_stats.get("statistics", []):
                    if s["type"] in STAT_MAP:
                        val = _stat_value(s["value"])
                        if val is None and s["type"] in ("Yellow Cards", "Red Cards"):
                            val = 0.0  # API kart yoksa null döndürüyor
                        row[STAT_MAP[s["type"]][side]] = val
            players = detail.get("players_slim") or {}
            for pre, team in (("h", th), ("a", ta)):
                ag_ = _team_player_aggs(players.get(str(team["id"]), []))
                for k in ("duel", "drb", "kp", "tkl", "rating"):
                    if k in ag_:
                        row[f"{pre}{k}"] = ag_[k]
                for pr in players.get(str(team["id"]), []):
                    p_min[team["id"]][pr["id"]] += pr["min"]
                    p_ga[team["id"]][pr["id"]] += pr["goals"] + pr["assists"]
            if "events_slim" in detail:
                for pre, team in (("h", th), ("a", ta)):
                    row[f"{pre}pen"] = sum(
                        1 for e in detail["events_slim"]
                        if e["team_id"] == team["id"] and (
                            (e["type"] == "Goal" and e["detail"] == "Penalty")
                            or e["detail"] == "Missed Penalty"))
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for _, (h, a) in STAT_MAP.items():
        for c in (h, a):
            if c not in df:
                df[c] = None
    return df.sort_values(["date", "home"]).reset_index(drop=True)


def convert_other() -> pd.DataFrame:
    """Kupa/Avrupa maçlarından takım-tarih listesi (yalnızca Süper Lig takımları)."""
    from teams import CANONICAL
    rows = []
    for op in CACHE.glob("*/other_*.json"):
        for f in json.loads(op.read_text(encoding="utf-8")):
            if f["status"] in DROP:
                continue
            dt = (pd.Timestamp(f["date"]).tz_convert("UTC") + pd.Timedelta(hours=3)).date()
            for t in (f["home"], f["away"]):
                n = normalize(t)
                if n in CANONICAL:
                    rows.append(dict(date=dt, team=n, comp=op.stem))
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["date", "team", "comp"])


def _round_no(r):
    if not r:
        return None
    tail = str(r).rsplit("-", 1)[-1].strip()
    return int(tail) if tail.isdigit() else None


def main():
    ap = argparse.ArgumentParser(description="API-Football'dan Süper Lig verisi")
    ap.add_argument("--sezon", type=int, nargs="*", default=[],
                    help="Sezon başlangıç yılı (2026 = 2026-27)")
    ap.add_argument("--max-istek", type=int, default=90, help="Bu çalıştırmada en fazla istek")
    ap.add_argument("--donustur", action="store_true", help="İndirmeden sadece CSV üret")
    ap.add_argument("--yenile", action="store_true", help="Fikstür listesini zorla yenile")
    a = ap.parse_args()

    if a.sezon and not a.donustur:
        cli = Client(load_key(), a.max_istek)
        for s in a.sezon:
            try:
                fetch_season(cli, s, force_list=a.yenile)
            except ApiError as e:
                print(f"  {s}: DURDU -> {e}")
                # Anahtar/limit hatasında diğer sezonları denemenin anlamı yok;
                # "plan" hatasında (sezon ücretsiz planda yok) sıradaki sezona geç
                if any(k in str(e) for k in ("istek sınırına", "anahtarı yok", "token", "requests")):
                    break
        left = f", günlük kalan: {cli.remaining_day}" if cli.remaining_day is not None else ""
        print(f"Kullanılan istek: {cli.used}{left}")

    df = convert()
    if df.empty:
        print("Önbellekte API verisi yok.")
        return
    out = ROOT / "data" / "api_matches.csv"
    df.to_csv(out, index=False)
    other = convert_other()
    other.to_csv(ROOT / "data" / "api_other_matches.csv", index=False)
    print(f"Kupa/Avrupa maçı (takım-tarih): {len(other)}")
    print(f"{len(df)} maç ({int(df.played.sum())} oynanmış) -> {out}")
    known = set(pd.read_csv(ROOT / "data" / "matches.csv").home) if (ROOT / "data" / "matches.csv").exists() else set()
    new = sorted((set(df.home) | set(df.away)) - known)
    if known and new:
        print("Uyarı: geçmiş veride olmayan takım adları (yeni takım ya da eşleştirme gerekli):",
              ", ".join(new))


if __name__ == "__main__":
    main()
