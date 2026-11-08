"""Nesine bülteninden iddaa oranları.

Yasal sitelerin (Nesine, iddaa.com, Bilyoner, Misli, Oley) standart oranları aynı
olduğu için tek kaynak yeterli. Bülten 5 dakika önbellekte tutulur.

    from nesine import find_odds
    find_odds("2026-11-08", "Çaykur Rizespor", "Erzurumspor")
    -> {"o1x2": (1.64, 3.3, 3.65), "oou": (1.48, 1.95), "mbs": 1, "saat": "13:30", ...}

Test:  python src/nesine.py
"""
import gzip
import json
import re
import time
import unicodedata
import urllib.request
from difflib import SequenceMatcher

import pandas as pd

URL = "https://bulten.nesine.com/api/bulten/getprebultenfull"
CACHE_SEC = 300
_cache = {"t": 0.0, "events": None}

# Takım adlarından atılacak ekler / sponsor adları
_DROP = {"fk", "sk", "as", "jk", "fc", "spor", "kulubu", "caykur", "rams", "corendon", "bitexen",
         "tumosan", "ikas", "hesap", "com", "net", "zecorner", "atakas", "mke", "yukatel"}


def norm(name):
    """'Çaykur Rizespor' -> 'rize', 'Erzurumspor FK' -> 'erzurum'"""
    s = unicodedata.normalize("NFKD", str(name).replace("ı", "i").replace("İ", "i"))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    words = [w for w in re.split(r"[^a-z0-9]+", s) if w and w not in _DROP]
    return " ".join(w[:-4] if w.endswith("spor") and len(w) > 6 else w for w in words)


# Alt yaş / kadın / B takımı işaretleri: bizim adda yoksa o maç başka takımındır
_SQUAD = re.compile(r"\b(u\d{2}|k|kadin|kadinlar|women|b|ii|res|reserves|gelisim|akademi)\b")


def _same(a, b):
    """Bizim ad (a) ile Nesine adı (b) ne kadar aynı takım: 0..1"""
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if set(_SQUAD.findall(b)) - set(_SQUAD.findall(a)):
        return 0.0  # örn. 'fenerbahce u19' asıl takım değil
    if a == b:
        return 1.0
    if re.search(rf"\b{re.escape(a)}\b", b) or re.search(rf"\b{re.escape(b)}\b", a):
        return 0.95  # 'basaksehir' ~ 'istanbul basaksehir'
    return SequenceMatcher(None, a, b).ratio()


def bulletin():
    """Futbol maçlarının listesi (önbellekli). Bağlantı koparsa 3 kez dener."""
    if _cache["events"] is not None and time.time() - _cache["t"] < CACHE_SEC:
        return _cache["events"]
    req = urllib.request.Request(URL, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130",
        "Accept-Encoding": "gzip", "Referer": "https://www.nesine.com/"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            break
        except OSError:
            if i == 2:
                raise
            time.sleep(1.5)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    ev = [e for e in json.loads(raw.decode("utf-8"))["sg"]["EA"] if e.get("GT") == 1 and e.get("HN")]
    _cache.update(t=time.time(), events=ev)
    return ev


def _market(e, t, st, line=0.0):
    for m in e.get("MA", []):
        if m["MT"] == t and m["MST"] == st and float(m.get("SOV") or 0) == line:
            o = {x["N"]: x["O"] for x in m.get("OCA", []) if isinstance(x.get("O"), (int, float)) and x["O"] > 1.0}
            return m, o
    return None, {}


last_error = None  # son bağlantı hatası (uygulama notunda gösterilir)


def find_odds(date, home, away):
    """Maçı bültende bulur; yoksa None. Ağ hatasında da None ve last_error dolar."""
    global last_error
    try:
        events = bulletin()
        last_error = None
    except Exception as ex:  # noqa: BLE001 - internet yoksa uygulama eski oranlara döner
        last_error = str(ex)
        return None
    d = pd.Timestamp(date).normalize()
    # gece yarısı civarı maçlarda tarih bir gün kayabilir
    days = {(d + pd.Timedelta(days=k)).strftime("%d.%m.%Y") for k in (-1, 0, 1)}
    best, score = None, 0.0
    for e in events:
        if e.get("D") not in days:
            continue
        s = min(_same(home, e["HN"]), _same(away, e["AN"]))
        if e.get("D") != d.strftime("%d.%m.%Y"):
            s -= 0.01  # aynı gün olan tercih edilir
        if s > score:
            best, score = e, s
    if best is None or score < 0.85:
        return None
    ms_m, ms = _market(best, 1, 1)
    _, au = _market(best, 2, 101, 2.5)
    return dict(
        o1x2=(ms[1], ms[2], ms[3]) if all(k in ms for k in (1, 2, 3)) else None,
        oou=(au[2], au[1]) if 1 in au and 2 in au else None,  # (Üst, Alt) - value_table sırası
        mbs=ms_m["MBS"] if ms_m else best.get("EMBS"),
        saat=best.get("T"), ad=f"{best['HN']} - {best['AN']}", mac_no=best.get("C"))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from features import load_matches
    m = load_matches()
    up = m[m.date >= pd.Timestamp.now().normalize()].sort_values("date").head(12)
    for r in up.itertuples():
        print(f"{r.date:%d.%m}  {r.home} - {r.away}  ->  {find_odds(r.date, r.home, r.away)}")
