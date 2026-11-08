"""livescoreFootball (worldcup26.ir) üzerinden güncel Süper Lig verisi.

Ücretsiz, anahtar gerektirmez (dakikada 120 istek sınırı). Kaynak:
https://github.com/rezarahiminia/livescoreFootball

Sağladıkları: tarih, skor, şut, isabetli şut, korner, topla oynama, faul ve
olaylardan (gol dakikaları, kartlar, penaltılar) türetilen ilk yarı skoru ve kart sayıları.
xG ve geçmiş sezonlar yok; o boşluğu API-Football doldurur.

Kullanım:
  python src/livescore_source.py            # indir + data/livescore_matches.csv üret
  python src/livescore_source.py --donustur # sadece önbellekten üret
Çıktı parse_openfootball.py tarafından otomatik birleştirilir
(öncelik: API-Football > livescore > openfootball).
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from teams import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "livescore"
BASE = "https://worldcup26.ir/get/soccer"
LEAGUE = "tur.1"
STAT_MAP = {"totalShots": "s", "shotsOnTarget": "st", "wonCorners": "c",
            "possessionPct": "poss", "foulsCommitted": "f"}


def _get(path, params):
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "superlig-tahmin/1.0"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if i == 2:
                raise
            time.sleep(5 * (i + 1))


def fetch():
    CACHE.mkdir(parents=True, exist_ok=True)
    events, page = [], 1
    while True:
        d = _get(f"{LEAGUE}/fixtures", {"status": "all", "limit": 100, "page": page})
        events += d.get("events", [])
        if page >= int(d.get("pageCount") or 1):
            break
        page += 1
        time.sleep(1)
    (CACHE / "fixtures.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
    print(f"livescore: {len(events)} maç indirildi")


def _minute(display: str) -> int | None:
    m = re.match(r"\s*(\d+)", display or "")
    return int(m.group(1)) if m else None


def _num(v):
    try:
        return float(str(v).rstrip("%"))
    except (TypeError, ValueError):
        return None


def convert() -> pd.DataFrame:
    p = CACHE / "fixtures.json"
    if not p.exists():
        return pd.DataFrame()
    rows, mismatch = [], 0
    for e in json.loads(p.read_text(encoding="utf-8")):
        comp = e["competitions"][0]
        st = comp["status"]["type"]
        if st["name"] in ("STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_ABANDONED"):
            continue
        played = bool(st.get("completed"))
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        h, a = sides["home"], sides["away"]
        dt = (pd.Timestamp(e["date"]).tz_convert("UTC") + pd.Timedelta(hours=3)).date()
        year = int(e["season"]["year"])
        row = dict(season=f"{year}-{(year + 1) % 100:02d}", date=dt, round=None,
                   home=normalize(h["team"]["displayName"]), away=normalize(a["team"]["displayName"]),
                   hg=int(h["score"]) if played else None, ag=int(a["score"]) if played else None,
                   hthg=None, htag=None, played=played)
        if played:
            for pre, c in (("h", h), ("a", a)):
                for s in c.get("statistics") or []:
                    if s["name"] in STAT_MAP:
                        key = STAT_MAP[s["name"]]
                        col = (pre + key) if pre == "h" else ("a" + key)
                        if key == "s":
                            col = "hs" if pre == "h" else "as"
                        row[col] = _num(s.get("displayValue"))
            det = comp.get("details") or []
            if det:
                ids = {h["team"]["id"]: "h", a["team"]["id"]: "a"}
                cnt = {k: 0 for k in ("hg_ev", "ag_ev", "hthg", "htag", "hy", "ay", "hr", "ar", "hpen", "apen")}
                for x in det:
                    side = ids.get(str((x.get("team") or {}).get("id")))
                    if side is None:
                        continue
                    t = (x.get("type") or {}).get("text", "")
                    mnt = _minute((x.get("clock") or {}).get("displayValue", ""))
                    if x.get("scoringPlay") and not x.get("shootout"):
                        cnt[f"{side}g_ev"] += 1
                        if mnt is not None and mnt <= 45:
                            cnt[f"ht{side}g" if False else ("hthg" if side == "h" else "htag")] += 1
                        if x.get("penaltyKick") or "Penalty" in t:
                            cnt[f"{side}pen"] += 1
                    if x.get("yellowCard"):
                        cnt[f"{side}y"] += 1
                    if x.get("redCard"):
                        cnt[f"{side}r"] += 1
                # Olaylardan sayılan gol gerçek skorla tutuyorsa ilk yarı skorunu kullan
                if cnt["hg_ev"] == row["hg"] and cnt["ag_ev"] == row["ag"]:
                    row["hthg"], row["htag"] = cnt["hthg"], cnt["htag"]
                    row["hpen"], row["apen"] = cnt["hpen"], cnt["apen"]
                else:
                    mismatch += 1
                for k in ("hy", "ay", "hr", "ar"):
                    row[k] = cnt[k]
            elif row["hg"] == 0 and row["ag"] == 0:
                row["hthg"] = row["htag"] = 0
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["date", "home"]).reset_index(drop=True)
    if mismatch:
        print(f"Uyarı: {mismatch} maçta olay verisi skorla tutmadı; ilk yarı skoru boş bırakıldı")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donustur", action="store_true")
    a = ap.parse_args()
    if not a.donustur:
        fetch()
    df = convert()
    if df.empty:
        print("livescore verisi yok")
        return
    out = ROOT / "data" / "livescore_matches.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} maç ({int(df.played.sum())} oynanmış), "
          f"{df.date.min()} - {df.date.max()} -> {out.name}")
    print("İlk yarı skoru olan:", int(df.hthg.notna().sum()), "| şut verisi olan:", int(df["hs"].notna().sum()) if "hs" in df else 0)


if __name__ == "__main__":
    main()
