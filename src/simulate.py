"""Monte Carlo sezon simülasyonu.

Mevcut puan tablosundan başlayıp kalan maçları model skor olasılıklarına göre
binlerce kez oynatır; şampiyonluk, ilk 2, Avrupa ve küme düşme olasılıklarını verir.

  python src/simulate.py                       # en güncel sezon, 10.000 simülasyon
  python src/simulate.py --sezon 2026-27 --n 20000 --dusen 3
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import load_matches  # noqa: E402
from models import score_matrix  # noqa: E402
from predict import predict  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sezon")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--dusen", type=int, default=3, help="Küme düşen takım sayısı")
    ap.add_argument("--avrupa", type=int, default=4, help="Avrupa kupalarına giden sıra sayısı")
    ap.add_argument("--tohum", type=int, default=42)
    a = ap.parse_args()

    m = load_matches()
    season = a.sezon or m.season.max()
    s = m[m.season == season]
    teams = sorted(set(s.home) | set(s.away))
    idx = {t: i for i, t in enumerate(teams)}
    played, rest = s[s.played], s[~s.played]
    print(f"{season}: {len(played)} maç oynandı, {len(rest)} maç simüle edilecek ({a.n} kez)")

    pts = np.zeros(len(teams))
    gd = np.zeros(len(teams))
    gf = np.zeros(len(teams))
    for r in played.itertuples(index=False):
        h, w = idx[r.home], idx[r.away]
        hg, ag = int(r.hg), int(r.ag)
        gf[h] += hg; gf[w] += ag; gd[h] += hg - ag; gd[w] += ag - hg
        pts[h] += 3 if hg > ag else 1 if hg == ag else 0
        pts[w] += 3 if ag > hg else 1 if hg == ag else 0

    rng = np.random.default_rng(a.tohum)
    P = np.tile(pts, (a.n, 1))
    GD = np.tile(gd, (a.n, 1))
    GF = np.tile(gf, (a.n, 1))
    if len(rest):
        pr = predict(rest[["season", "date", "home", "away"]], m, weather=False)
        for r in pr.itertuples(index=False):
            matrix = score_matrix(r.lh, r.la, r.dc_rho)
            mat = matrix.ravel()
            k = rng.choice(len(mat), size=a.n, p=mat / mat.sum())
            hg, ag = np.unravel_index(k, matrix.shape)
            h, w = idx[r.home], idx[r.away]
            P[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            P[:, w] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
            GD[:, h] += hg - ag; GD[:, w] += ag - hg
            GF[:, h] += hg; GF[:, w] += ag

    # Sıralama: puan, averaj, atılan gol (+ küçük rastgele eşitlik bozucu)
    score = P * 1e6 + GD * 1e3 + GF + rng.random(P.shape) * 0.1
    rank = (-score).argsort(1).argsort(1) + 1  # 1 = lider
    T = len(teams)
    out = pd.DataFrame({
        "takım": teams,
        "puan_şimdi": pts.astype(int),
        "beklenen_puan": P.mean(0).round(1),
        "ort_sıra": rank.mean(0).round(1),
        "şampiyon_%": (100 * (rank == 1).mean(0)).round(1),
        "ilk2_%": (100 * (rank <= 2).mean(0)).round(1),
        f"ilk{a.avrupa}_%": (100 * (rank <= a.avrupa).mean(0)).round(1),
        "düşme_%": (100 * (rank > T - a.dusen).mean(0)).round(1),
    }).sort_values("beklenen_puan", ascending=False)
    REP = ROOT / "reports"
    REP.mkdir(exist_ok=True)
    out.to_csv(REP / f"simulation_{season}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
