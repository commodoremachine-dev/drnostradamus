"""ESPN'in herkese açık (anahtarsız) API'sinden Süper Lig verisi. Kişisel kullanım içindir.

İki aşama:
  1) Fikstür/sonuç: yıl başına tek istek  (scoreboard?dates=YYYY)  -> tüm maçlar
  2) Maç detayı:   maç başına tek istek  (summary?event=ID)      -> maç öncesi bahis
     oranları (1X2, 2.5 alt/üst), ilk yarı skoru, kart/penaltı olayları, takım
     istatistikleri (2025-26'dan itibaren), ilk 11'ler
Detaylar data/raw/espn/summary/ altında önbelleklenir; tekrar çalıştırınca yalnızca
eksik ve yeni biten maçlar indirilir. Yarıda kesilirse kaldığı yerden devam eder.

Kullanım:
  python src/espn_source.py                    # 2017'den bugüne her şey (~20-25 dk ilk sefer)
  python src/espn_source.py --yillar 2026      # sadece 2026 takvim yılı
  python src/espn_source.py --detaysiz         # sadece sonuçlar (hızlı)
  python src/espn_source.py --donustur         # indirmeden CSV üret
Çıktı: data/espn_matches.csv (parse_openfootball.py otomatik birleştirir)
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import date
from fractions import Fraction
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from teams import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "espn"
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/tur.1"
STAT_MAP = {"totalShots": "s", "shotsOnTarget": "st", "wonCorners": "c",
            "possessionPct": "poss", "foulsCommitted": "f"}
DONE = {"STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_AET", "STATUS_FINAL_PEN"}
SKIP = {"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_ABANDONED"}
DELAY = 0.3
WORKERS = 4  # aynı anda en fazla 4 istek


def _get(url):
    import urllib.error
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (personal research)",
                                               "Accept-Encoding": "identity"})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (400, 404) or i == 3:  # olmayan kayıt: tekrar denemenin anlamı yok
                raise
            time.sleep(3 * (i + 1))
        except Exception:
            if i == 3:
                raise
            time.sleep(3 * (i + 1))


# ------------------------------------------------------------------ indirme
def fetch_years(years):
    d = CACHE / "scoreboard"
    d.mkdir(parents=True, exist_ok=True)
    for y in years:
        p = d / f"{y}.json"
        # geçmiş yıllar bir kez, içinde bulunulan yıl her seferinde yenilenir
        if p.exists() and y < date.today().year:
            continue
        data = _get(f"{BASE}/scoreboard?dates={y}&limit=1000")
        ev = data.get("events", [])
        p.write_text(json.dumps(ev, ensure_ascii=False), encoding="utf-8")
        print(f"  {y}: {len(ev)} maç")
        time.sleep(DELAY)


def all_events():
    out = {}
    for p in sorted((CACHE / "scoreboard").glob("*.json")):
        for e in json.loads(p.read_text(encoding="utf-8")):
            out[e["id"]] = e
    return list(out.values())


def _slim_summary(s: dict) -> dict:
    comp = s["header"]["competitions"][0]
    teams = {c["homeAway"]: {"id": c["team"]["id"], "name": c["team"]["displayName"],
                             "score": c.get("score"),
                             "ls": [x.get("displayValue") for x in c.get("linescores") or []]}
             for c in comp["competitors"]}
    odds = []
    for o in s.get("odds") or []:
        to = ((o.get("bettingOdds") or {}).get("teamOdds")) or {}
        odds.append({
            "provider": (o.get("provider") or {}).get("name"),
            "pm_home": (to.get("preMatchFullTimeResultHome") or {}).get("value"),
            "pm_draw": (to.get("preMatchFullTimeResultDraw") or {}).get("value"),
            "pm_away": (to.get("preMatchFullTimeResultAway") or {}).get("value"),
            "pm_ou_line": (to.get("preMatchOverUnderHandicap") or {}).get("value"),
            "pm_over": (to.get("preMatchGoalLineOver") or {}).get("value"),
            "pm_under": (to.get("preMatchGoalLineUnder") or {}).get("value"),
            "ml_home": (o.get("homeTeamOdds") or {}).get("moneyLine"),
            "ml_away": (o.get("awayTeamOdds") or {}).get("moneyLine"),
            "ml_draw": (o.get("drawOdds") or {}).get("moneyLine"),
            "ou": o.get("overUnder"), "over_odds": o.get("overOdds"), "under_odds": o.get("underOdds"),
        })
    events = [{"type": (k.get("type") or {}).get("text"), "team": (k.get("team") or {}).get("id"),
               "period": (k.get("period") or {}).get("number"), "scoring": k.get("scoringPlay")}
              for k in s.get("keyEvents") or []]
    stats = {}
    for t in (s.get("boxscore") or {}).get("teams") or []:
        stats[t["team"]["id"]] = {x.get("name"): x.get("displayValue") for x in t.get("statistics") or []}
    starters = {}
    for r in s.get("rosters") or []:
        starters[r["team"]["id"]] = [p["athlete"]["id"] for p in r.get("roster") or [] if p.get("starter")]
    return {"teams": teams, "status": comp["status"]["type"]["name"], "odds": odds,
            "events": events, "stats": stats, "starters": starters,
            "referee": next((o.get("displayName") for o in (s.get("gameInfo") or {}).get("officials") or []
                             if "Referee" in ((o.get("position") or {}).get("displayName") or "Referee")), None)}


def fetch_summaries(events, limit=None):
    d = CACHE / "summary"
    d.mkdir(parents=True, exist_ok=True)
    todo = [e for e in events
            if e["competitions"][0]["status"]["type"]["name"] in DONE and not (d / f"{e['id']}.json").exists()]
    # yaklaşan 10 gün içindeki maçlar: oranlar için her seferinde yenilenir
    soon = [e for e in events if e["competitions"][0]["status"]["type"]["name"] not in DONE | SKIP
            and 0 <= (pd.Timestamp(e["date"]).tz_convert(None) - pd.Timestamp.now()).days <= 10]
    todo = (todo + soon)[:limit] if limit else todo + soon
    print(f"Maç detayı indirilecek: {len(todo)}", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def one(e):
        try:
            s = _get(f"{BASE}/summary?event={e['id']}")
            (d / f"{e['id']}.json").write_text(json.dumps(_slim_summary(s), ensure_ascii=False), encoding="utf-8")
            return True
        except Exception as x:
            return f"{e['id']}: {x}"
        finally:
            time.sleep(DELAY)

    fails = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            if r is not True:
                fails.append(r)
            if i % 250 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    if fails:
        print(f"  {len(fails)} maç alınamadı (ör. {fails[0]})")


OTHER_COMPS = ["uefa.champions", "uefa.europa", "uefa.europa.conf",
               "uefa.champions_qual", "uefa.europa_qual", "uefa.europa.conf_qual"]


def fetch_other(years):
    """Türk takımlarının Avrupa kupası maç tarihleri (yorgunluk değişkenleri için)."""
    d = CACHE / "other"
    d.mkdir(parents=True, exist_ok=True)
    for comp in OTHER_COMPS:
        for y in years:
            p = d / f"{comp}_{y}.json"
            if p.exists() and y < date.today().year:
                continue
            try:
                ev = _get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{comp}/scoreboard?dates={y}&limit=1000").get("events", [])
            except Exception:
                continue
            slim = [{"date": e["date"], "status": e["competitions"][0]["status"]["type"]["name"],
                     "teams": [c["team"]["displayName"] for c in e["competitions"][0]["competitors"]]} for e in ev]
            p.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
            time.sleep(DELAY)


def convert_other() -> pd.DataFrame:
    from teams import CANONICAL
    rows = []
    for p in (CACHE / "other").glob("*.json"):
        comp = p.stem.rsplit("_", 1)[0]
        for e in json.loads(p.read_text(encoding="utf-8")):
            if e["status"] in SKIP:
                continue
            dt = (pd.Timestamp(e["date"]).tz_convert("UTC") + pd.Timedelta(hours=3)).date()
            for t in e["teams"]:
                n = normalize(t)
                if n in CANONICAL:
                    rows.append(dict(date=dt, team=n, comp=comp))
    return pd.DataFrame(rows, columns=["date", "team", "comp"]).drop_duplicates()


# ------------------------------------------------------------------ dönüştürme
def _frac(v):
    try:
        return float(Fraction(str(v))) + 1.0
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _ml(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return 1 + v / 100 if v > 0 else 1 + 100 / abs(v) if v < 0 else None


def _implied(*dec):
    if any(x is None or x <= 1 for x in dec):
        return None
    inv = [1 / x for x in dec]
    s = sum(inv)
    return [x / s for x in inv]  # bahis şirketi marjı çıkarılmış olasılık


def _odds_features(odds):
    out = {}
    for o in odds:  # önce Bet365 maç öncesi kesirli oranlar, yoksa moneyline
        dec = (_frac(o["pm_home"]), _frac(o["pm_draw"]), _frac(o["pm_away"]))
        p = _implied(*dec)
        if p is None:
            dec = (_ml(o["ml_home"]), _ml(o["ml_draw"]), _ml(o["ml_away"]))
            p = _implied(*dec)
        if p and "odds_ph" not in out:
            out.update(odds_ph=p[0], odds_pd=p[1], odds_pa=p[2],
                       # ham ondalık oranlar (value bet ve kâr/zarar hesabı için)
                       dec_h=round(dec[0], 3), dec_d=round(dec[1], 3), dec_a=round(dec[2], 3),
                       odds_book=o.get("provider"))
        if "odds_pover25" not in out:
            if str(o.get("pm_ou_line")) == "2.5":
                dov, dun = _frac(o["pm_over"]), _frac(o["pm_under"])
            elif o.get("ou") == 2.5:
                dov, dun = _ml(o.get("over_odds")), _ml(o.get("under_odds"))
            else:
                dov = dun = None
            q = _implied(dov, dun) if dov and dun else None
            if q:
                out.update(odds_pover25=q[0], dec_over25=round(dov, 3), dec_under25=round(dun, 3))
    return out


def _num(v):
    try:
        return float(str(v).rstrip("%"))
    except (TypeError, ValueError):
        return None


def convert() -> pd.DataFrame:
    rows = []
    sdir = CACHE / "summary"
    for e in all_events():
        comp = e["competitions"][0]
        status = comp["status"]["type"]["name"]
        if status in SKIP:
            continue
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        h, a = sides["home"], sides["away"]
        played = status in DONE
        year = int(e["season"]["year"])
        dt = (pd.Timestamp(e["date"]).tz_convert("UTC") + pd.Timedelta(hours=3)).date()
        row = dict(season=f"{year}-{(year + 1) % 100:02d}", date=dt, round=None,
                   home=normalize(h["team"]["displayName"]), away=normalize(a["team"]["displayName"]),
                   hg=int(h["score"]) if played else None, ag=int(a["score"]) if played else None,
                   hthg=None, htag=None, played=played)
        sp = sdir / f"{e['id']}.json"
        s = None
        if sp.exists():
            try:
                s = json.loads(sp.read_text(encoding="utf-8"))
            except ValueError:
                sp.unlink()  # bozuk dosya: bir sonraki çalıştırmada yeniden indirilir
        if s is not None:
            th, ta = s["teams"]["home"], s["teams"]["away"]
            if played and len(th["ls"]) >= 1 and len(ta["ls"]) >= 1:
                try:
                    row["hthg"], row["htag"] = int(th["ls"][0]), int(ta["ls"][0])
                except (TypeError, ValueError):
                    pass
            row.update(_odds_features(s["odds"]))
            if played:
                cnt = {"hy": 0, "ay": 0, "hr": 0, "ar": 0, "hpen": 0, "apen": 0}
                for ev in s["events"]:
                    side = "h" if ev["team"] == th["id"] else "a" if ev["team"] == ta["id"] else None
                    t = ev["type"] or ""
                    if side is None:
                        continue
                    if t == "Yellow Card":
                        cnt[f"{side}y"] += 1
                    elif t in ("Red Card", "Yellow Red Card"):
                        cnt[f"{side}r"] += 1
                    elif "Penalty" in t and ev.get("scoring"):
                        cnt[f"{side}pen"] += 1
                if s["events"]:
                    row.update(cnt)
                for pre, t in (("h", th), ("a", ta)):
                    st = s["stats"].get(t["id"]) or {}
                    for k, short in STAT_MAP.items():
                        if k in st and _num(st[k]) is not None:
                            col = ("hs" if pre == "h" else "as") if short == "s" else pre + short
                            row[col] = _num(st[k])
            if s.get("referee"):
                row["referee"] = s["referee"]
            st = s.get("starters") or {}
            for pre, t in (("h", th), ("a", ta)):
                xi = st.get(t["id"]) or []
                if len(xi) == 11:
                    row[f"{pre}_xi"] = "|".join(map(str, xi))
        elif played:
            for pre, c in (("h", h), ("a", a)):  # detay yoksa scoreboard istatistikleri
                for x in c.get("statistics") or []:
                    if x["name"] in STAT_MAP and _num(x.get("displayValue")) is not None:
                        short = STAT_MAP[x["name"]]
                        col = ("hs" if pre == "h" else "as") if short == "s" else pre + short
                        row[col] = _num(x["displayValue"])
        if played and row["hthg"] is None and row["hg"] == 0 and row["ag"] == 0:
            row["hthg"] = row["htag"] = 0
        rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates(["season", "home", "away"], keep="last")
    return df.sort_values(["date", "home"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yillar", type=int, nargs="*", default=list(range(2017, date.today().year + 1)))
    ap.add_argument("--detaysiz", action="store_true")
    ap.add_argument("--donustur", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if not a.donustur:
        print("Sonuçlar indiriliyor...")
        fetch_years(a.yillar)
        fetch_other(a.yillar)
        if not a.detaysiz:
            fetch_summaries(all_events(), a.limit)
    oth = convert_other()
    oth.to_csv(ROOT / "data" / "espn_other_matches.csv", index=False)
    print(f"Avrupa kupası maçı (takım-tarih): {len(oth)}")
    df = convert()
    out = ROOT / "data" / "espn_matches.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} maç ({int(df.played.sum())} oynanmış) -> {out.name}")
    for s, g in df[df.played].groupby("season"):
        print(f"  {s}: {len(g)} maç | ilk yarı {g.hthg.notna().sum()} | oran "
              f"{g['odds_ph'].notna().sum() if 'odds_ph' in g else 0} | şut {g['hs'].notna().sum() if 'hs' in g else 0}")


if __name__ == "__main__":
    main()
