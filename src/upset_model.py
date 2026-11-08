"""Sürpriz modeli: favorinin kaybetme olasılığını özel olarak tahmin eder.

Favori = ana modelin (oran varsa bahis piyasasının) kazanmasını daha olası gördüğü taraf.
Hedef = favori kaybetti mi (berabere sürpriz sayılmaz).

Girdiler favori bakış açısından hazırlanır: ana model ve piyasa olasılıkları, ikisi
arasındaki fark, güç farkları, dinlenme, deplasman, derbi, sezon başı, geçen sezon, eksikler...
Eğitim walk-forward: her test sezonu yalnızca önceki sezonların verisiyle eğitilir.

Çalıştır:  python src/upset_model.py      (önce backtest.py çalışmış olmalı)
Çıktılar:  reports/upset_report.csv, reports/upset_levels.json (alarm seviyelerinin geçmiş isabeti)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TEST_SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]

# favori/zayıf olarak çevrilecek ev-deplasman değişken çiftleri
PAIRS = ["elo", "ppg5", "ppg10", "gf_ewm", "ga_ewm", "gf_venue_ewm", "ga_venue_ewm", "over_perf10",
         "season_ppg", "season_gdpg", "rank", "rest_days", "games_21d", "other_comp_14d",
         "prev_ppg", "prev_rank", "season_games", "home_adv", "missing", "missing_min_share",
         "xpts_ewm", "hxg_for_ewm", "hxg_ag_ewm", "hs_for_ewm", "hst_for_ewm", "hst_ag_ewm",
         "lead_lost_rate", "comeback_rate", "unbeaten_streak", "winless_streak", "big", "promoted",
         "dc_att", "dc_def", "cs_rate10", "fts_rate10",
         "xi_continuity", "xi_regulars", "regulars_missing", "xi_new_faces"]


def _side_col(df, side, name):
    for c in (f"{side}_{name}", f"elo_{side}" if name == "elo" else None,
              f"dc_{name.split('_')[1]}_{side}" if name.startswith("dc_") else None):
        if c and c in df.columns:
            return df[c]
    return None


def favorite_frame(df: pd.DataFrame, ph, pd_, pa) -> pd.DataFrame:
    """Her maçı favori bakış açısına çevirir."""
    fav_home = ph >= pa
    X = pd.DataFrame(index=df.index)
    X["p_fav"] = np.where(fav_home, ph, pa)
    X["p_dog"] = np.where(fav_home, pa, ph)
    X["p_draw"] = pd_
    X["fav_margin"] = X.p_fav - X.p_dog
    X["fav_is_home"] = fav_home.astype(int)
    if "model_ph" in df.columns:
        mph, mpa = df.model_ph.values, df.model_pa.values
        X["model_p_fav"] = np.where(fav_home, mph, mpa)
        X["model_p_dog"] = np.where(fav_home, mpa, mph)
        X["model_minus_ref_dog"] = X.model_p_dog - X.p_dog  # >0: model zayıf tarafa referanstan fazla şans veriyor
    if "odds_ph" in df.columns:
        oph, opa = df.odds_ph.values, df.odds_pa.values
        X["mkt_p_fav"] = np.where(fav_home, oph, opa)
        X["mkt_p_dog"] = np.where(fav_home, opa, oph)
        X["mkt_p_draw"] = df.odds_pd.values
        X["model_vs_mkt_fav"] = X.p_fav - X.mkt_p_fav  # >0: model favoriye piyasadan fazla güveniyor
        X["mkt_over25"] = df.get("odds_pover25")
    for name in PAIRS:
        h, a = _side_col(df, "home", name), _side_col(df, "away", name)
        if h is None or a is None:
            continue
        fv, dv = np.where(fav_home, h, a), np.where(fav_home, a, h)
        X[f"fav_{name}"], X[f"dog_{name}"] = fv, dv
        X[f"gap_{name}"] = fv.astype(float) - dv.astype(float)
    for c in ["is_derby", "season_progress", "month", "away_travel_km", "same_city", "h2h_gd_home",
              "w_rain_mm", "w_wind_max", "w_temp_max", "ref_home_win_rate", "pi_exp_gd"]:
        if c in df.columns:
            v = df[c].values.astype(float)
            # ev sahibine göre olanları favoriye göre çevir
            X[c] = np.where(fav_home, v, -v) if c in ("h2h_gd_home", "pi_exp_gd") else v
    return X


def make_clf():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=0.02, max_iter=3000))


def make_gbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.02, num_leaves=4, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                              verbose=-1, random_state=0)


def load_data():
    oof = pd.read_csv(REPORTS / "oof_probs.csv", parse_dates=["date"])
    tab = pd.read_csv(ROOT / "data" / "model_table.csv", parse_dates=["date"])
    d = oof.merge(tab.drop(columns=["season", "hg", "ag"]), on=["date", "home", "away"], how="left")
    # Referans olasılık: piyasa varsa piyasa, yoksa model (favoriyi belirlemek için).
    # Böylece "sürpriz" = piyasanın (yoksa modelin) favorisinin kaybetmesi.
    has = d.mkt_h.notna() if "mkt_h" in d else pd.Series(False, index=d.index)
    ph = np.where(has, d.get("mkt_h"), d.final_h).astype(float)
    pd_ = np.where(has, d.get("mkt_d"), d.final_d).astype(float)
    pa = np.where(has, d.get("mkt_a"), d.final_a).astype(float)
    d["model_ph"], d["model_pa"] = d.final_h, d.final_a
    X = favorite_frame(d, ph, pd_, pa)
    fav_home = X.fav_is_home.values == 1
    res = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
    y = np.where(fav_home, res == 2, res == 0).astype(int)  # favori kaybetti
    return d, X, y


def evaluate_scores(y, s, name, base_rate):
    from sklearn.metrics import roc_auc_score
    out = {"model": name, "n": len(y), "auc": roc_auc_score(y, s)}
    order = np.argsort(-s)
    for q in (0.05, 0.10, 0.20):
        k = max(1, int(len(y) * q))
        out[f"isabet_ilk%{int(q * 100)}"] = y[order[:k]].mean()
    out["genel_surpriz_orani"] = base_rate
    return out


def main():
    d, X, y = load_data()
    cols = list(X.columns)
    rows, preds = [], []
    for season in TEST_SEASONS:
        tr, te = d.season < season, d.season == season
        if te.sum() == 0 or tr.sum() < 300:
            continue
        p_lr = make_clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        p_gb = make_gbm().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        preds.append(pd.DataFrame({
            "season": season, "date": d.date[te].values, "home": d.home[te].values, "away": d.away[te].values,
            "y": y[te], "ana_model": X.model_p_dog[te].values if "model_p_dog" in X else X.p_dog[te].values,
            "piyasa": np.where(d.mkt_h[te].notna(), X.p_dog[te].values, np.nan),
            "surpriz_lr": p_lr, "surpriz_gbm": p_gb, "surpriz_ort": (p_lr + p_gb) / 2}))
    P = pd.concat(preds, ignore_index=True)
    base = P.y.mean()
    for c, n in [("ana_model", "Ana model (zayıf tarafın olasılığı)"),
                 ("piyasa", "Bahis piyasası (zayıf tarafın olasılığı)"),
                 ("surpriz_lr", "Sürpriz modeli (lojistik)"),
                 ("surpriz_gbm", "Sürpriz modeli (LightGBM)"),
                 ("surpriz_ort", "Sürpriz modeli (ortalama)")]:
        m = P[c].notna()
        if m.sum() < 100:
            continue
        rows.append(evaluate_scores(P.y[m].values, P[c][m].values, n + ("" if m.all() else f" [oranlı {m.sum()} maç]"), P.y[m].mean()))
    R = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    R.to_csv(REPORTS / "upset_report.csv", index=False)
    P.to_csv(REPORTS / "upset_predictions.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Test: {len(P)} maç, sürpriz oranı %{100 * base:.1f}\n")
    print(R.round(3).to_string(index=False))

    # Alarm seviyeleri: sürpriz olasılığına göre geçmişteki gerçek isabet
    s = P.surpriz_gbm.values  # testte en iyi ayırt eden: LightGBM
    cuts = np.quantile(s, [0.6, 0.85, 0.95])
    lv = np.digitize(s, cuts)
    names = ["düşük", "orta", "yüksek", "çok yüksek"]
    levels = []
    print("\nSürpriz alarm seviyeleri (test sezonlarındaki gerçek sonuçlar):")
    for i, n in enumerate(names):
        m = lv == i
        levels.append({"seviye": n, "alt_sinir": float(([0] + list(cuts))[i]), "mac": int(m.sum()),
                       "gercek_surpriz": float(P.y[m].mean()), "maclarin_payi": float(m.mean())})
        print(f"  {n:11s} {m.sum():4d} maç (%{100 * m.mean():4.1f}) | favori kaybetti: %{100 * P.y[m].mean():.1f}")
    # Son model: tüm veriyle eğit (tahmin aracı kullanır)
    import pickle
    final = {"lr": make_clf().fit(X, y), "gbm": make_gbm().fit(X, y), "cols": cols, "levels": levels}
    with open(REPORTS / "upset_model.pkl", "wb") as f:
        pickle.dump(final, f)
    (REPORTS / "upset_levels.json").write_text(json.dumps(levels, ensure_ascii=False, indent=2), encoding="utf-8")


def predict_upset(target: pd.DataFrame, ph, pd_, pa):
    """Tahmin aracı için: maç(lar)ın sürpriz olasılığı ve alarm seviyesi."""
    import pickle
    p = REPORTS / "upset_model.pkl"
    if not p.exists():
        return None
    m = pickle.load(open(p, "rb"))
    X = favorite_frame(target.reset_index(drop=True), np.asarray(ph), np.asarray(pd_), np.asarray(pa))
    for c in m["cols"]:
        if c not in X:
            X[c] = np.nan
    X = X[m["cols"]]
    s = m["gbm"].predict_proba(X)[:, 1]
    out = []
    for v in s:
        lv = [l for l in m["levels"] if v >= l["alt_sinir"]][-1]
        out.append({"surpriz_olasiligi": float(v), "seviye": lv["seviye"],
                    "gecmis_isabet": lv["gercek_surpriz"], "fav_is_home": None})
    for o, fh in zip(out, X.fav_is_home):
        o["fav_is_home"] = bool(fh)
    return out


if __name__ == "__main__":
    main()
