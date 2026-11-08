"""openfootball (Football.TXT) Süper Lig dosyalarını tek bir CSV'ye çevirir.

Çıktı: data/matches.csv
Sütunlar: season, date, round, home, away, hg, ag, hthg, htag, played
"""
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from teams import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), start=1)}

DATE_RE = re.compile(r"^\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
ROUND_RE = re.compile(r"^\s*▪\s*(?:Matchday|Round|Regular Season -)?\s*(\d+)")
TIME = r"(?:\d{1,2}[:.]\d{2})?"
SCORE = r"(\d+)-(\d+)(?:\s*\((\d+)-(\d+)\))?"
# Format A: "20:30  Ev  v Deplasman  2-1 (1-0)"
FMT_A = re.compile(rf"^\s*{TIME}\s+(.+?)\s+v\s+(.+?)\s+{SCORE}")
# Format B: "21:30   Ev  2-2 (0-0)  Deplasman"
FMT_B = re.compile(rf"^\s*{TIME}\s+(.+?)\s+{SCORE}\s+(.+?)\s*$")
# Oynanmamış maç: "Ev   v   Deplasman"
FMT_U = re.compile(rf"^\s*{TIME}\s+(.+?)\s+v\s+(.+?)\s*$")


def season_year(month: int, start_year: int) -> int:
    return start_year if month >= 7 else start_year + 1


def parse_file(path: Path) -> list[dict]:
    start_year = int(path.name[:4])
    season = path.name[:7]
    rows, cur_date, rnd = [], None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "=", "(")) or s.endswith(("')", "';")):
            continue
        m = ROUND_RE.match(line)
        if m:
            rnd = int(m.group(1))
            continue
        m = DATE_RE.match(line)
        if m:
            mon = MONTHS[m.group(2)]
            cur_date = date(season_year(mon, start_year), mon, int(m.group(3)))
            continue
        if cur_date is None:
            continue
        rec = None
        m = FMT_A.match(line)
        if m:
            rec = (m.group(1), m.group(2), *m.groups()[2:6])
        else:
            m = FMT_B.match(line)
            if m:
                rec = (m.group(1), m.group(6), *m.groups()[1:5])
            else:
                m = FMT_U.match(line)
                if m:
                    rec = (m.group(1), m.group(2), None, None, None, None)
        if rec is None:
            continue
        home, away, hg, ag, hthg, htag = rec
        rows.append(dict(
            season=season, date=cur_date, round=rnd,
            home=normalize(home), away=normalize(away),
            hg=_int(hg), ag=_int(ag), hthg=_int(hthg), htag=_int(htag),
            played=hg is not None,
        ))
    return rows


def _int(x):
    return int(x) if x is not None else None


EXTRA_COLS = ["hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar"]


def merge_extra_stats(df: pd.DataFrame) -> pd.DataFrame:
    """data/raw/extra/*.csv içindeki maç istatistiklerini (şut, korner, kart) ekler.

    Beklenen sütunlar: date (YYYY-MM-DD veya DD/MM/YYYY), home, away ve şunlardan
    istenenler: hs, as, hst, ast, hc, ac, hy, ay, hr, ar
    (football-data.co.uk formatındaki Date/HomeTeam/AwayTeam/HS/AS/... adları da kabul edilir.)
    """
    files = sorted((RAW / "extra").glob("*.csv"))
    if not files:
        return df
    parts = []
    for f in files:
        e = pd.read_csv(f)
        e = e.rename(columns={"Date": "date", "HomeTeam": "home", "AwayTeam": "away"})
        e.columns = [c.lower() if c.lower() in EXTRA_COLS else c for c in e.columns]
        e["date"] = pd.to_datetime(e["date"], dayfirst=True, format="mixed").dt.date
        e["home"] = e["home"].map(normalize)
        e["away"] = e["away"].map(normalize)
        parts.append(e[["date", "home", "away"] + [c for c in EXTRA_COLS if c in e.columns]])
    extra = pd.concat(parts).drop_duplicates(["date", "home", "away"]).rename(columns={"as": "as_"})
    out = df.merge(extra, on=["date", "home", "away"], how="left")
    print(f"Ek istatistik eşleşen maç: {out['hs'].notna().sum() if 'hs' in out else 0}")
    return out


MIN_SEASON = "2017-18"  # ESPN'in 2016-17 verisi yarım (yalnızca Ocak sonrası)


def merge_source(df: pd.DataFrame, csv_name: str, label: str) -> pd.DataFrame:
    """Bir kaynağın maçlarını mevcut tabloya işler (sonra gelen kaynak önceliklidir).

    Bir sezonda her ev-deplasman eşleşmesi tek kez oynandığı için (sezon, ev, deplasman)
    anahtarıyla eşleştirilir; ertelenen maçların tarihleri de böylece düzelir.
    Kaynakta olmayan sezonlar/maçlar olduğu gibi kalır.
    """
    p = ROOT / "data" / csv_name
    if not p.exists():
        return df
    src = pd.read_csv(p, parse_dates=["date"])
    src["date"] = src.date.dt.date
    src = src.rename(columns={"as": "as_"})
    src = src[src.season >= MIN_SEASON]
    key = ["season", "home", "away"]
    both = df[df.played].merge(src[src.played], on=key, suffixes=("_old", "_new"))
    bad = both[(both.hg_old != both.hg_new) | (both.ag_old != both.ag_new)]
    if len(bad):
        print(f"Uyarı: {len(bad)} maçta skorlar kaynaklar arasında farklı ({label} kullanıldı)")
    known = set(map(tuple, df[key].values))
    drop = set(map(tuple, src[key].values))
    # Eski kaynakta olup yenide olmayan ek istatistikleri koru (örn. livescore kartları)
    old = df[df[key].apply(tuple, axis=1).isin(drop)].set_index(key)
    src = src.set_index(key)
    for c in old.columns:
        if c in src.columns:
            src[c] = src[c].where(src[c].notna(), old[c].reindex(src.index))
        elif c not in ("date", "round"):
            src[c] = old[c].reindex(src.index)
    src = src.reset_index()
    keep = df[~df[key].apply(tuple, axis=1).isin(drop)]
    print(f"{label}: {int(src.played.sum())} oynanmış maç, {len(drop - known)} yeni eşleşme")
    return pd.concat([keep, src], ignore_index=True).sort_values(["date", "home"]).reset_index(drop=True)


def merge_livescore(df):
    return merge_source(df, "livescore_matches.csv", "livescore")


def merge_espn(df):
    return merge_source(df, "espn_matches.csv", "ESPN")


def merge_api(df: pd.DataFrame) -> pd.DataFrame:
    """data/api_matches.csv varsa, kapsadığı sezonlarda openfootball yerine onu kullanır.

    API verisi daha güncel ve şut/xG/kart/eksik oyuncu bilgisi içerir.
    """
    p = ROOT / "data" / "api_matches.csv"
    if not p.exists():
        return df
    api = pd.read_csv(p, parse_dates=["date"])
    api["date"] = api.date.dt.date
    api = api.rename(columns={"as": "as_"}).drop(columns=["fixture_id"], errors="ignore")
    seasons = sorted(api.season.unique())
    keep = df[~df.season.isin(seasons)]
    print(f"API verisi kullanılan sezonlar: {', '.join(seasons)}")
    return pd.concat([keep, api], ignore_index=True).sort_values(["date", "home"]).reset_index(drop=True)


def main():
    rows = []
    for p in sorted(RAW.glob("*_tr1.txt")):
        r = parse_file(p)
        print(f"{p.name}: {len(r)} maç ({sum(x['played'] for x in r)} oynanmış)")
        rows += r
    df = pd.DataFrame(rows).sort_values(["date", "home"]).reset_index(drop=True)
    # Kaynak 0-0 biten maçlarda ilk yarı skorunu yazmıyor
    zero = (df.hg == 0) & (df.ag == 0)
    df.loc[zero, ["hthg", "htag"]] = 0
    df = merge_extra_stats(df)
    df = merge_livescore(df)
    df = merge_espn(df)
    df = merge_api(df)
    for c in ["hg", "ag", "hthg", "htag", "round"]:
        df[c] = df[c].astype("Int64")
    out = ROOT / "data" / "matches.csv"
    df.to_csv(out, index=False)
    print(f"Toplam {len(df)} satır -> {out}")


if __name__ == "__main__":
    main()
