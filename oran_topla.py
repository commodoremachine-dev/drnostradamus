"""Günün iddaa futbol bültenini Nesine'den çeker.

Yasal sitelerin (Nesine, iddaa.com, Bilyoner, Misli, Oley) standart oranları aynı
olduğu için tek kaynak yeterli. Nesine en çok pazarı veriyor.

Çıktılar (betiğin yanına):
  oranlar.csv      her maç için tek satır: MBS, MS 1/X/2, 2,5 Alt/Üst, KG Var/Yok
  tum_oranlar.csv  her maçın her pazarı ve seçeneği (uzun liste)

Kullanım:  python oran_topla.py
"""
import csv
import gzip
import json
import os
import urllib.request
from datetime import datetime

URL = "https://bulten.nesine.com/api/bulten/getprebultenfull"
OUT = os.path.dirname(os.path.abspath(__file__))

# Resmî pazar kodları: (tür, alt tür, çizgi)
MS = (1, 1, 0.0)
AU25 = (2, 101, 2.5)
KG = (2, 89, 0.0)


def fetch():
    req = urllib.request.Request(URL, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130",
        "Accept-Encoding": "gzip", "Referer": "https://www.nesine.com/"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))["sg"]["EA"]


def odd(v):
    return v if isinstance(v, (int, float)) and v > 1.0 else ""  # 1.00 = kapalı


def main():
    events = [e for e in fetch() if e.get("GT") == 1 and e.get("HN")]  # GT 1 = futbol
    events.sort(key=lambda e: e["ESD"])
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(os.path.join(OUT, "oranlar.csv"), "w", newline="", encoding="utf-8-sig") as f, \
         open(os.path.join(OUT, "tum_oranlar.csv"), "w", newline="", encoding="utf-8-sig") as g:
        w = csv.writer(f, delimiter=";")
        w.writerow(["cekim", "mac_no", "tarih", "saat", "ev", "deplasman", "MBS",
                    "MS1", "MSX", "MS2", "Alt2.5", "Ust2.5", "KGVar", "KGYok"])
        wl = csv.writer(g, delimiter=";")
        wl.writerow(["cekim", "mac_no", "ev", "deplasman", "tur", "alt_tur", "cizgi", "MBS", "secenek", "oran"])

        for e in events:
            mk = {}
            for m in e.get("MA", []):
                key = (m["MT"], m["MST"], float(m.get("SOV") or 0))
                mk[key] = m
                for o in m.get("OCA", []):
                    if odd(o.get("O")):
                        wl.writerow([stamp, e["C"], e["HN"], e["AN"], m["MT"], m["MST"],
                                     m.get("SOV") or "", m.get("MBS"), o["N"], o["O"]])

            def o(key, n):
                m = mk.get(key)
                if not m:
                    return ""
                return next((odd(x.get("O")) for x in m["OCA"] if x["N"] == n), "")

            mbs = mk[MS]["MBS"] if MS in mk else e.get("EMBS", "")
            w.writerow([stamp, e["C"], e["D"], e["T"], e["HN"], e["AN"], mbs,
                        o(MS, 1), o(MS, 2), o(MS, 3), o(AU25, 1), o(AU25, 2), o(KG, 1), o(KG, 2)])

    tek = sum(1 for e in events for m in e.get("MA", []) if m["MT"] == 1 and m["MST"] == 1 and m["MBS"] == 1)
    print(f"{len(events)} futbol maçı kaydedildi ({tek} tanesi tek maç oynanabilir, MBS 1).")
    print(f"-> {os.path.join(OUT, 'oranlar.csv')}")


if __name__ == "__main__":
    main()
