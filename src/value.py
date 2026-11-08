"""Value bet (değerli oran) analizi.

Value bet: bahis şirketinin oranı, olayın gerçek olasılığına göre olması gerekenden yüksekse.
  Beklenen değer (EV) = p × oran − 1     (EV > 0 ise oran "değerli")
  Adil oran           = 1 / p

İki olasılık kaynağı gösterilir:
  - Adil (piyasa + model): keskin piyasanın (Bet365/ESPN) marjı temizlenmiş olasılığı modelle
    birleştirilip kalibre edilir. Kendi bahis şirketinizin oranını buna göre değerlendirin
    (gerçek hayatta value bet böyle bulunur: oran karşılaştırması).
  - Model (bağımsız): yalnızca istatistik modeli. Bilgi amaçlı; geçmiş testte zarar ettirdi.

  python src/value.py test                                   # geçmiş sezonlarda kâr/zarar testi
  python src/value.py tahmin Galatasaray Fenerbahçe --oran 1.85 3.60 4.20 [--ust 1.70 2.10]
  python src/value.py hafta                                  # 7 gün içindeki maçlar (ESPN oranlarıyla)

UYARI: Kişisel analiz aracıdır. Geçmiş test sonucu gelecekte kâr garantisi değildir.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODEL_ROW = "SEÇİLEN + kalibrasyon"   # backtest'teki piyasadan bağımsız kalibre model
FAIR_ROW = "NİHAİ: piyasa + model"     # adil olasılık: piyasa + model, kalibre
SOURCES = {"Model (bağımsız)": MODEL_ROW, "Adil (piyasa+model)": FAIR_ROW}
THRESHOLDS = [0.0, 0.02, 0.05, 0.10]
HOLDOUT = ["2024-25", "2025-26", "2026-27"]
VALUE_MIN_EV = 0.05  # value işareti için adil orana göre en az %5 fazla


def ev(p, dec):
    return p * dec - 1


# ------------------------------------------------------------------ geriye dönük test
def load_bets(row=MODEL_ROW) -> pd.DataFrame:
    P = pd.read_csv(REPORTS / "backtest_predictions.csv")
    P = P[P.model == row]
    M = pd.read_csv(ROOT / "data" / "matches.csv")
    cols = ["season", "home", "away", "dec_h", "dec_d", "dec_a", "dec_over25", "dec_under25"]
    D = P.merge(M[[c for c in cols if c in M.columns]], on=["season", "home", "away"], how="inner")
    rows = []
    for r in D.itertuples(index=False):
        tot = r.hg + r.ag
        for out, p, dec, won in (("1", r.p_home, r.dec_h, r.hg > r.ag), ("X", r.p_draw, r.dec_d, r.hg == r.ag),
                                 ("2", r.p_away, r.dec_a, r.hg < r.ag),
                                 ("2.5 Üst", r.p_over25, getattr(r, "dec_over25", np.nan), tot > 2),
                                 ("2.5 Alt", 1 - r.p_over25, getattr(r, "dec_under25", np.nan), tot <= 2)):
            if pd.isna(dec) or dec <= 1:
                continue
            rows.append(dict(season=r.season, date=r.date, home=r.home, away=r.away, secim=out,
                             pazar="1X2" if out in "1X2" else "2.5", p=p, oran=dec, ev=ev(p, dec),
                             kazandi=bool(won), kar=(dec - 1) if won else -1.0))
    return pd.DataFrame(rows)


def roi_ci(kar: np.ndarray, n_boot=4000, seed=0):
    if len(kar) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(kar, size=(n_boot, len(kar)), replace=True).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def summarize(B: pd.DataFrame, label: str) -> dict:
    lo, hi = roi_ci(B.kar.values)
    return {"strateji": label, "bahis": len(B), "isabet_%": 100 * B.kazandi.mean() if len(B) else np.nan,
            "ort_oran": B.oran.mean(), "kar_birim": B.kar.sum(), "ROI_%": 100 * B.kar.mean() if len(B) else np.nan,
            "ROI_alt_%": 100 * lo, "ROI_ust_%": 100 * hi}


def cmd_test():
    out, n_match = [], 0
    for src_name, row in SOURCES.items():
        B = load_bets(row)
        if B.empty:
            sys.exit("Oranlı test maçı yok. Önce espn_source.py ve backtest.py çalıştırın.")
        n_match = B[["season", "home", "away"]].drop_duplicates().shape[0]
        for scope, Bs in (("TÜM TEST", B), ("GÖRÜLMEMİŞ", B[B.season.isin(HOLDOUT)])):
            if src_name == "Model (bağımsız)":
                out.append(summarize(Bs[Bs.pazar == "1X2"], f"[{scope}] Referans: 1X2'de her şeye oyna"))
                fav = Bs[Bs.pazar == "1X2"].loc[
                    lambda d: d.groupby(["season", "home", "away"]).oran.transform("min") == d.oran]
                out.append(summarize(fav, f"[{scope}] Referans: hep piyasa favorisi"))
            for pazar in ("1X2", "2.5"):
                for th in THRESHOLDS:
                    sel = Bs[(Bs.pazar == pazar) & (Bs.ev > th)]
                    # maç başına en fazla bir bahis (en yüksek EV)
                    sel = sel.sort_values("ev", ascending=False).drop_duplicates(["season", "home", "away", "pazar"])
                    out.append(summarize(sel, f"[{scope}] {src_name} | {pazar} EV > %{int(th * 100)}"))
    R = pd.DataFrame(out)
    REPORTS.mkdir(exist_ok=True)
    R.to_csv(REPORTS / "value_backtest.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"Oranlı test maçı: {n_match}  (Bet365/ESPN oranları, 1 birim sabit bahis; "
          f"ROI = bahis başına ortalama kâr, %95 güven aralığıyla)\n")
    print(R.round(2).to_string(index=False))
    B = load_bets(MODEL_ROW)
    v = B[(B.ev > 0.05)].sort_values("ev", ascending=False).drop_duplicates(["season", "home", "away", "pazar"])
    print("\nModel (bağımsız), EV > %5 bahislerin seçime göre sonucu (tüm test):")
    print(v.groupby("secim").agg(bahis=("kar", "size"), isabet=("kazandi", "mean"), ROI=("kar", "mean"),
                                 ort_oran=("oran", "mean")).round(3).to_string())


# ------------------------------------------------------------------ canlı kullanım
def value_table(r, odds_1x2=None, odds_ou=None) -> pd.DataFrame:
    """Bir maç için adil/model olasılığı, adil oran, verilen oran ve EV tablosu."""
    from predict import match_markets
    _, mk = match_markets(r, use_market=False)   # piyasadan bağımsız model olasılığı
    _, fk = match_markets(r, use_market=True)    # adil olasılık (piyasa oranı varsa piyasa + model)
    has_mkt = not pd.isna(getattr(r, "odds_ph", np.nan))
    if odds_1x2 is None and not pd.isna(getattr(r, "dec_h", np.nan)):
        odds_1x2 = (r.dec_h, r.dec_d, r.dec_a)
    if odds_ou is None and not pd.isna(getattr(r, "dec_over25", np.nan)):
        odds_ou = (r.dec_over25, r.dec_under25)
    rows = []
    for name, p, pf, dec in (("1", mk["p_home"], fk["p_home"], odds_1x2[0] if odds_1x2 else None),
                             ("X", mk["p_draw"], fk["p_draw"], odds_1x2[1] if odds_1x2 else None),
                             ("2", mk["p_away"], fk["p_away"], odds_1x2[2] if odds_1x2 else None),
                             ("2.5 Üst", mk["p_over25"], fk["p_over25"], odds_ou[0] if odds_ou else None),
                             ("2.5 Alt", 1 - mk["p_over25"], 1 - fk["p_over25"], odds_ou[1] if odds_ou else None)):
        rows.append(dict(secim=name, p_model=p, p_adil=pf, adil_oran=1 / pf if pf > 0 else np.nan, oran=dec,
                         ev_adil=ev(pf, dec) if dec else np.nan, ev_model=ev(p, dec) if dec else np.nan,
                         piyasa_var=has_mkt))
    return pd.DataFrame(rows)


def print_value(r, tbl: pd.DataFrame):
    print(f"\n{r.home} - {r.away}  ({pd.Timestamp(r.date):%d.%m.%Y})")
    src = "piyasa + model" if tbl.piyasa_var.iloc[0] else "yalnızca model (piyasa oranı yok, güvenilirlik düşük)"
    print(f"  Adil olasılık kaynağı: {src}")
    print(f"  {'Seçim':8s} {'Adil %':>7s} {'Adil oran':>10s} {'Oran':>6s} {'EV(adil)':>9s} {'Model %':>8s} {'EV(model)':>10s}")
    for t in tbl.itertuples():
        o = f"{t.oran:.2f}" if t.oran else "  -"
        ea = f"{t.ev_adil * 100:+.1f}%" if not pd.isna(t.ev_adil) else "   -"
        em = f"{t.ev_model * 100:+.1f}%" if not pd.isna(t.ev_model) else "   -"
        flag = "  <- VALUE" if not pd.isna(t.ev_adil) and t.ev_adil > VALUE_MIN_EV else ""
        print(f"  {t.secim:8s} {t.p_adil * 100:6.1f}% {t.adil_oran:10.2f} {o:>6s} {ea:>9s} "
              f"{t.p_model * 100:7.1f}% {em:>10s}{flag}")
    if tbl.oran.isna().all():
        print("  (Oran yok: --oran EV BERABERLIK DEPLASMAN [--ust UST ALT] ile kendi bahis şirketinizin oranlarını girin)")


def _footer():
    p = REPORTS / "value_backtest.csv"
    print(f"\nNasıl okunur: EV(adil) = oranınızın adil orana göre beklenen kâr/zararı. "
          f"VALUE işareti: EV(adil) > %{int(VALUE_MIN_EV * 100)}.")
    if p.exists():
        R = pd.read_csv(p)
        for key, label in (("Adil (piyasa+model) | 1X2 EV > %5", "adil olasılık"),
                           ("Model (bağımsız) | 1X2 EV > %5", "model olasılığı")):
            row = R[R.strateji.str.contains("GÖRÜLMEMİŞ", regex=False) & R.strateji.str.contains(key, regex=False)]
            if len(row):
                x = row.iloc[0]
                print(f"Geçmiş test, Bet365'e karşı ({label}, EV > %5, görülmemiş sezonlar): {int(x.bahis)} bahis, "
                      f"ROI %{x['ROI_%']:+.1f} (%95 aralık %{x['ROI_alt_%']:+.1f} / %{x['ROI_ust_%']:+.1f})")
    print("Uyarı: Bet365'e karşı kanıtlanmış bir avantaj yok; EV(model) geçmişte zarar ettirdi. Value ancak sizin "
          "oranınız adil oranın belirgin üstündeyse anlamlıdır. Kâr garantisi yoktur; sorumlu oynayın.")


def cmd_tahmin(a):
    from features import load_matches
    from predict import make_fixture, predict, resolve_team
    m = load_matches()
    known = sorted(set(m.home) | set(m.away))
    home, away = resolve_team(a.ev, known), resolve_team(a.deplasman, known)
    d = pd.Timestamp(a.tarih)
    # Fikstürde varsa (ESPN oranlarıyla) onu kullan
    hit = m[(m.home == home) & (m.away == away) & (~m.played)]
    fx = hit[["season", "date", "home", "away"]].head(1) if len(hit) else make_fixture(home, away, d)
    r = next(predict(fx, m).itertuples(index=False))
    if len(hit):
        r = r._replace(**{c: hit.iloc[0][c] for c in ("dec_h", "dec_d", "dec_a", "dec_over25", "dec_under25")
                          if c in hit.columns and c in r._fields})
    print_value(r, value_table(r, tuple(a.oran) if a.oran else None, tuple(a.ust) if a.ust else None))
    _footer()


def cmd_hafta():
    from features import load_matches
    from predict import predict
    m = load_matches()
    now = pd.Timestamp.now().normalize()
    fx = m[(~m.played) & m.date.between(now, now + pd.Timedelta(days=7))]
    if fx.empty:
        sys.exit("7 gün içinde maç yok.")
    res = predict(fx[["season", "date", "home", "away"]], m)
    extra_cols = [c for c in ("dec_h", "dec_d", "dec_a", "dec_over25", "dec_under25") if c in m.columns]
    res = res.drop(columns=[c for c in extra_cols if c in res.columns])
    res = res.merge(m[["season", "home", "away"] + extra_cols], on=["season", "home", "away"], how="left")
    for r in res.sort_values("date").itertuples(index=False):
        print_value(r, value_table(r))
    _footer()


def main():
    ap = argparse.ArgumentParser(description="Value bet analizi")
    sub = ap.add_subparsers(dest="komut", required=True)
    sub.add_parser("test")
    sub.add_parser("hafta")
    t = sub.add_parser("tahmin")
    t.add_argument("ev")
    t.add_argument("deplasman")
    t.add_argument("--tarih", default=str(pd.Timestamp.now().date()))
    t.add_argument("--oran", type=float, nargs=3, metavar=("EV", "BERABERLIK", "DEPLASMAN"))
    t.add_argument("--ust", type=float, nargs=2, metavar=("UST", "ALT"))
    a = ap.parse_args()
    if a.komut == "test":
        cmd_test()
    elif a.komut == "hafta":
        cmd_hafta()
    else:
        cmd_tahmin(a)


if __name__ == "__main__":
    main()
