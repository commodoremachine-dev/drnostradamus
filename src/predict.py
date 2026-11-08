"""Maç skoru tahmini.

Örnekler:
  python src/predict.py Galatasaray Fenerbahçe
  python src/predict.py Galatasaray Fenerbahçe --tarih 2026-11-08
  python src/predict.py Samsunspor Beşiktaş --ev-carpan 0.9      # ev sahibinde önemli eksikler
  python src/predict.py --fikstur 2026-11-06 2026-11-10          # veri setindeki fikstürler
  python src/predict.py Galatasaray Fenerbahçe --yz               # + Claude yapay zekâ analizi

--ev-carpan / --dep-carpan: modelin göremediği etkenler için beklenen golü elle çarpar.
Not: önce `python src/backtest.py` çalıştırılmış olmalı (eğitim tablosunu üretir).
"""
import argparse
import difflib
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest import FINAL_SOURCES, dc_row, dense_columns, feature_columns, make_glm, make_lgbm  # noqa: E402
from context_data import build_context, load_other_dates  # noqa: E402
from features import build_features, load_matches  # noqa: E402
from models import DixonColes, markets, score_matrix  # noqa: E402
from teams import normalize  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def season_of(d: pd.Timestamp) -> str:
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def resolve_team(name: str, known: list[str]) -> str:
    n = normalize(name)
    for k in known:
        if k.casefold() == n.casefold():
            return k
    match = difflib.get_close_matches(n, known, n=1, cutoff=0.6)
    if match:
        return match[0]
    sys.exit(f"Takım bulunamadı: {name}. Bilinen takımlar: {', '.join(sorted(known))}")


def make_fixture(home, away, d: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame([dict(season=season_of(d), date=d, round=pd.NA, home=home, away=away,
                              hg=pd.NA, ag=pd.NA, hthg=pd.NA, htag=pd.NA, played=False)])


_MODEL_CACHE = {}


def _trained_models(train, cols, stamp):
    """Gol modellerini eğitir; aynı eğitim tablosu için oturum boyunca önbellekte tutar."""
    key = (stamp, tuple(cols))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        makers = {"glm": (make_glm, dense_columns(train, cols)), "lgb": (make_lgbm, cols)}
        out = {}
        for s in FINAL_SOURCES:
            factory, cc = makers[s]
            out[s] = (factory().fit(train[cc], train.hg.astype(float)),
                      factory().fit(train[cc], train.ag.astype(float)), cc)
        _MODEL_CACHE[key] = out
    return _MODEL_CACHE[key]


def predict(fixtures: pd.DataFrame, matches: pd.DataFrame, ev_carpan=1.0, dep_carpan=1.0,
            weather=True) -> pd.DataFrame:
    """Fikstürler için tüm değişkenleri ve son modelin beklenen gollerini döndürür."""
    table_path = ROOT / "data" / "model_table.csv"
    if not table_path.exists():
        sys.exit("Önce eğitim tablosunu üretin: python src/backtest.py")
    train = pd.read_csv(table_path, parse_dates=["date"])
    train = train[train.played]

    key = ["date", "home", "away"]
    extra = fixtures[~fixtures.set_index(key).index.isin(matches.set_index(key).index)]
    allm = pd.concat([matches, extra], ignore_index=True).sort_values(["date", "home"]).reset_index(drop=True)
    ctx = build_context(allm, weather=weather)
    feats = build_features(allm, load_other_dates(), ctx)
    target = feats.merge(fixtures[key], on=key)

    rows = []
    for d, grp in target.groupby("date"):
        dc = DixonColes().fit(matches[matches.date < d], d)
        rows += [dc_row(dc, r) for r in grp.itertuples(index=False)]
    target = target.merge(pd.DataFrame(rows), on=key)

    cols = feature_columns(train)
    for c in cols:
        if c not in target:
            target[c] = np.nan
    models = _trained_models(train, cols, table_path.stat().st_mtime)
    lhs, las = [], []
    for s in FINAL_SOURCES:
        mh, ma, cc = models[s]
        lhs.append(np.log(mh.predict(target[cc])))
        las.append(np.log(ma.predict(target[cc])))
    target["lh"] = np.exp(np.mean(lhs, 0)) * ev_carpan
    target["la"] = np.exp(np.mean(las, 0)) * dep_carpan
    return target


def last_result_date(matches, team):
    m = matches[matches.played & ((matches.home == team) | (matches.away == team))]
    return m.date.max()


def _fmt(v, f="{:.2f}"):
    return "-" if v is None or pd.isna(v) else f.format(v)


def match_markets(r, use_market=True):
    """Kalibre edilmiş skor matrisi ve pazarlar (1X2 kalibrasyonu backtest'te öğrenildi)."""
    import json
    from backtest import apply_calibration
    from stacking import rescale_matrix
    from backtest import apply_blend
    m = score_matrix(r.lh, r.la, r.dc_rho)
    raw = markets(m)
    P = np.array([[raw["p_home"], raw["p_draw"], raw["p_away"]]])
    cal_p, blend_p = ROOT / "reports" / "calibration.json", ROOT / "reports" / "blend.json"
    target = apply_calibration(P, json.loads(cal_p.read_text(encoding="utf-8"))) if cal_p.exists() else P
    odds = np.array([[getattr(r, "odds_ph", np.nan), getattr(r, "odds_pd", np.nan), getattr(r, "odds_pa", np.nan)]],
                    dtype=float)
    if use_market and blend_p.exists() and not np.isnan(odds).any():
        # Oran varsa: piyasa + model birleşimi (geçmiş testte en iyi sonuç)
        target = apply_blend(P, odds, json.loads(blend_p.read_text(encoding="utf-8")), target)
    return rescale_matrix(m, target[0]), markets(rescale_matrix(m, target[0]))


def upset_info(r, mk=None):
    from upset_model import predict_upset
    # sürpriz modeli: favori piyasaya göre (oran yoksa modele göre); model görüşü ayrıca girdi
    mk = markets(score_matrix(r.lh, r.la, r.dc_rho))
    t = pd.DataFrame([r._asdict()]).assign(model_ph=mk["p_home"], model_pa=mk["p_away"])
    if not pd.isna(getattr(r, "odds_ph", np.nan)):
        ref = (r.odds_ph, r.odds_pd, r.odds_pa)
    else:
        ref = (mk["p_home"], mk["p_draw"], mk["p_away"])
    res = predict_upset(t, [ref[0]], [ref[1]], [ref[2]])
    return res[0] if res else None


def print_prediction(r, matches, detail=True):
    m, mk = match_markets(r)
    print(f"\n{'=' * 62}\n{r.home}  -  {r.away}   ({r.date:%d.%m.%Y})\n{'=' * 62}")
    print(f"Beklenen gol:  {r.home} {r.lh:.2f}   |   {r.away} {r.la:.2f}")
    print(f"1 / X / 2:     %{mk['p_home'] * 100:.1f}  /  %{mk['p_draw'] * 100:.1f}  /  %{mk['p_away'] * 100:.1f}")
    print(f"2.5 Üst:       %{mk['p_over25'] * 100:.1f}     KG Var: %{mk['p_btts'] * 100:.1f}")
    print("En olası skorlar: " + ", ".join(f"{s} (%{p * 100:.1f})" for s, p in mk["top_scores"]))
    if not pd.isna(getattr(r, "odds_ph", np.nan)):
        print(f"Bahis piyasası: %{r.odds_ph * 100:.1f}  /  %{r.odds_pd * 100:.1f}  /  %{r.odds_pa * 100:.1f}")
    u = upset_info(r, mk)
    if u:
        fav = r.home if u["fav_is_home"] else r.away
        print(f"SÜRPRİZ ALARMI: {u['seviye'].upper()} — {fav} kaybetme olasılığı %{u['surpriz_olasiligi'] * 100:.1f} "
              f"(bu seviyede geçmişte favori %{u['gecmis_isabet'] * 100:.0f} kaybetti)")
    if not detail:
        return
    print("\nSkor tablosu (%)   satır = ev golü, sütun = deplasman golü")
    print("      " + "".join(f"{j:>6d}" for j in range(6)))
    for i in range(6):
        print(f"{i:>4d}  " + "".join(f"{m[i, j] * 100:6.1f}" for j in range(6)))
    g = lambda k: getattr(r, k, np.nan)  # noqa: E731
    print("\nÖne çıkan değişkenler (ev | deplasman):")
    print(f"  Elo: {g('elo_home'):.0f} | {g('elo_away'):.0f}    Pi-rating: {_fmt(g('pi_home'))} | {_fmt(g('pi_away'))}")
    print(f"  Son 5 maç puan ort.: {_fmt(g('home_ppg5'))} | {_fmt(g('away_ppg5'))}    "
          f"İç saha/deplasman puan farkı: {_fmt(g('home_home_adv'))} | {_fmt(g('away_home_adv'))}")
    print(f"  Gol ort. attı/yedi: {_fmt(g('home_gf_ewm'))}/{_fmt(g('home_ga_ewm'))} | "
          f"{_fmt(g('away_gf_ewm'))}/{_fmt(g('away_ga_ewm'))}")
    if not pd.isna(g("home_hxg_for_ewm")):
        print(f"  xG/xGA: {_fmt(g('home_hxg_for_ewm'))}/{_fmt(g('home_hxg_ag_ewm'))} | "
              f"{_fmt(g('away_hxg_for_ewm'))}/{_fmt(g('away_hxg_ag_ewm'))}    "
              f"xPTS: {_fmt(g('home_xpts_ewm'))} | {_fmt(g('away_xpts_ewm'))}")
    print(f"  Dinlenme günü: {_fmt(g('home_rest_days'), '{:.0f}')} | {_fmt(g('away_rest_days'), '{:.0f}')}    "
          f"Deplasman yolu: {_fmt(g('away_travel_km'), '{:.0f}')} km")
    if not pd.isna(g("w_temp_max")):
        print(f"  Hava: {g('w_temp_max'):.0f}°C, yağış {g('w_rain_mm'):.1f} mm, rüzgar {g('w_wind_max'):.0f} km/s")
    if not pd.isna(g("home_missing")):
        print(f"  Eksik oyuncu: {g('home_missing'):.0f} | {g('away_missing'):.0f}")
    h2h = "yok" if pd.isna(g("h2h_gd_home")) else f"{g('h2h_gd_home'):+.2f} (son {int(g('h2h_n'))} maç)"
    print(f"  İkili rekabet (ev sahibine göre ort. gol farkı): {h2h}    Derbi: {'evet' if g('is_derby') else 'hayır'}")
    for t in (r.home, r.away):
        last = last_result_date(matches, t)
        if pd.notna(last) and (r.date - last).days > 21:
            print(f"  ! {t} için en son sonuç {last:%d.%m.%Y} tarihli; form verisi eski.")


def main():
    ap = argparse.ArgumentParser(description="Süper Lig skor tahmini")
    ap.add_argument("ev", nargs="?")
    ap.add_argument("deplasman", nargs="?")
    ap.add_argument("--tarih", default=str(date.today()))
    ap.add_argument("--ev-carpan", type=float, default=1.0)
    ap.add_argument("--dep-carpan", type=float, default=1.0)
    ap.add_argument("--fikstur", nargs=2, metavar=("BASLANGIC", "BITIS"))
    ap.add_argument("--yz", action="store_true", help="Claude ile yapay zekâ analizi ekle")
    a = ap.parse_args()

    matches = load_matches()
    known = sorted(set(matches.home) | set(matches.away))
    if a.fikstur:
        s, e = pd.Timestamp(a.fikstur[0]), pd.Timestamp(a.fikstur[1])
        fx = matches[(~matches.played) & matches.date.between(s, e)][["season", "date", "home", "away"]]
        if fx.empty:
            sys.exit("Bu tarih aralığında veri setinde oynanmamış maç yok.")
    else:
        if not (a.ev and a.deplasman):
            ap.error("İki takım adı veya --fikstur verin.")
        fx = make_fixture(resolve_team(a.ev, known), resolve_team(a.deplasman, known), pd.Timestamp(a.tarih))
    res = predict(fx, matches, a.ev_carpan, a.dep_carpan)
    for r in res.sort_values("date").itertuples(index=False):
        print_prediction(r, matches, detail=not a.fikstur or a.yz)
        if a.yz:
            from ai_analyst import analyze, print_analysis
            print_analysis(analyze(r, matches))


if __name__ == "__main__":
    main()
