"""Geriye dönük test (walk-forward): her maç yalnızca geçmiş veriyle tahmin edilir.

Çalıştır:  python src/backtest.py
Çıktılar:  data/model_table.csv (eğitim tablosu), reports/backtest_summary.csv,
           reports/backtest_predictions.csv, reports/feature_effects.csv,
           reports/stack_params.json (tahminde kullanılan birleştirme ağırlıkları)

Optimize model: 5 kaynağın (Dixon-Coles, Poisson GLM, LightGBM, lojistik regresyon,
LightGBM sınıflandırıcı) 1/X/2 olasılıklarını birleştirir. Ağırlıklar ve kalibrasyon her
test sezonu için yalnızca ÖNCEKİ sezonların tahminleri üzerinde öğrenilir.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from context_data import build_context, load_other_dates  # noqa: E402
from features import build_features, load_matches  # noqa: E402
from models import DixonColes, evaluate, markets, score_matrix  # noqa: E402
from stacking import apply_stack, fit_stack, rescale_matrix  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIRST_DC_SEASON = "2018-19"   # 2017-18 yalnızca ısınma için
STACK_ONLY_SEASONS = ["2021-22", "2022-23"]  # sadece birleştirme ağırlıklarını öğrenmek için
TEST_SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
REFIT_DAYS = 4

NON_FEATURES = {"season", "date", "round", "home", "away", "hg", "ag", "played", "referee"}


def dixon_coles_walkforward(matches: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Her maç için o tarihten önceki veriyle kurulmuş Dixon-Coles tahmini."""
    rows, model, last_fit = [], None, None
    target = feats[feats.season >= FIRST_DC_SEASON]
    for r in target.itertuples(index=False):
        if model is None or (r.date - last_fit).days >= REFIT_DAYS:
            model = DixonColes().fit(matches[matches.date < r.date], r.date)
            last_fit = r.date
        rows.append(dc_row(model, r))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "dc_walkforward.csv", index=False)
    return out


def dc_row(model, r):
    lh, la = model.lambdas(r.home, r.away, bool(r.home_promoted), bool(r.away_promoted))
    ah, dh = model.strength(r.home, bool(r.home_promoted))
    aa, da = model.strength(r.away, bool(r.away_promoted))
    return dict(date=r.date, home=r.home, away=r.away, dc_lh=lh, dc_la=la,
                dc_rho=model.rho, dc_att_home=ah, dc_def_home=dh,
                dc_att_away=aa, dc_def_away=da, dc_log_lh=np.log(lh), dc_log_la=np.log(la))


def feature_columns(df):
    # Bahis oranları gol modellerine girmez (eski sezonlarda yarısı eksik); en sonda piyasa
    # olasılıklarıyla ayrıca birleştirilir
    return [c for c in df.columns if c not in NON_FEATURES and c != "dc_rho"
            and not c.startswith("odds_") and c != "has_odds"
            and pd.api.types.is_numeric_dtype(df[c])]


def dense_columns(train, cols, max_missing=0.2):
    return [c for c in cols if train[c].isna().mean() <= max_missing]


def make_glm():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # keep_empty_features: eski sezonlarda hiç olmayan (xG gibi) sütunlar atılmasın
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         PoissonRegressor(alpha=3.0, max_iter=3000))  # alpha 2023-24'te seçildi


def make_lgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective="poisson", n_estimators=150, learning_rate=0.02,
                             num_leaves=4, min_child_samples=40, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                             verbose=-1, random_state=0)


def make_lr():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=0.01, max_iter=3000))


def make_lgbc():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=150, learning_rate=0.02,
                              num_leaves=4, min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.6, reg_lambda=5.0, verbose=-1, random_state=0)


def outcome(hg, ag):
    hg, ag = np.asarray(hg, float), np.asarray(ag, float)
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def probs_from_lambdas(lh, la, rho):
    P = np.zeros((len(lh), 3))
    for i in range(len(lh)):
        mk = markets(score_matrix(lh[i], la[i], rho[i]))
        P[i] = mk["p_home"], mk["p_draw"], mk["p_away"]
    return P


def base_predictions(train: pd.DataFrame, target: pd.DataFrame, cols: list) -> dict:
    """Tüm temel modelleri train üzerinde eğitip target için tahmin üretir."""
    rho = target.dc_rho.values
    lam = {"dc": (target.dc_lh.values, target.dc_la.values)}
    # GLM eksik veriye duyarlı: eğitim verisinde %20'den fazla boş olan sütunları kullanmaz
    # (LightGBM boşlukları kendisi yönetir, tüm sütunları kullanır)
    glm_cols = dense_columns(train, cols)
    for name, factory, cc in (("glm", make_glm, glm_cols), ("lgb", make_lgbm, cols)):
        mh = factory().fit(train[cc], train.hg.astype(float))
        ma = factory().fit(train[cc], train.ag.astype(float))
        lam[name] = (mh.predict(target[cc]), ma.predict(target[cc]))
    P = {k: probs_from_lambdas(v[0], v[1], rho) for k, v in lam.items()}
    y = outcome(train.hg, train.ag)
    for name, factory, cc in (("lr", make_lr, glm_cols), ("lgbc", make_lgbc, cols)):
        clf = factory().fit(train[cc], y)
        P[name] = clf.predict_proba(target[cc])
    # Skor pazarları için gol beklentisi: üç modelin geometrik ortalaması
    lh = np.exp(np.mean([np.log(v[0]) for v in lam.values()], 0))
    la = np.exp(np.mean([np.log(v[1]) for v in lam.values()], 0))
    return {"lam": lam, "P": P, "lh": lh, "la": la, "rho": rho}


# Son model: 2023-24 doğrulama sezonunda 31 birleşim arasından en yüksek doğrulukla seçildi
# (Poisson GLM + LightGBM gol modelleri, beklenen gollerin geometrik ortalaması).
FINAL_SOURCES = ["glm", "lgb"]
FINAL_NAME = "SEÇİLEN: GLM + LightGBM"


CAL_NAME = "SEÇİLEN + kalibrasyon"
BLEND_NAME = "NİHAİ: piyasa + model"


def fit_blend(prior: pd.DataFrame):
    """Oranı olan geçmiş maçlarda piyasa ve model olasılıklarının en iyi birleşimini öğrenir."""
    q = prior.dropna(subset=["mkt_h"]) if "mkt_h" in prior else prior.iloc[0:0]
    if len(q) < 200:
        return None
    # 'lgb' anahtarı piyasa için, 'dc' model için kullanılır (stacking kaynak adları)
    return fit_stack({"dc": q[["final_h", "final_d", "final_a"]].values,
                      "lgb": q[["mkt_h", "mkt_d", "mkt_a"]].values}, outcome(q.hg, q.ag), l2=0.0)


def apply_blend(P_model, P_mkt, blend, P_fallback):
    ok = ~np.isnan(P_mkt).any(1)
    out = P_fallback.copy()
    if ok.any():
        out[ok] = apply_stack({"dc": P_model[ok], "lgb": P_mkt[ok]}, blend)
    return out


def fit_calibration(prior: pd.DataFrame) -> dict:
    """Seçilen modelin geçmiş tahminlerinden sıcaklık (keskinlik) ve beraberlik ayarı öğrenir."""
    P = prior[["final_h", "final_d", "final_a"]].values
    return fit_stack({"dc": P}, outcome(prior.hg, prior.ag), l2=0.0)


def apply_calibration(P: np.ndarray, cal: dict) -> np.ndarray:
    return apply_stack({"dc": P}, cal)


def final_lambdas(b):
    lh = np.exp(np.mean([np.log(b["lam"][s][0]) for s in FINAL_SOURCES], 0))
    la = np.exp(np.mean([np.log(b["lam"][s][1]) for s in FINAL_SOURCES], 0))
    return lh, la


def final_rows(df, lh, la, rho, P_final, name):
    recs = []
    for i, r in enumerate(df.itertuples(index=False)):
        m = score_matrix(lh[i], la[i], rho[i])
        if P_final is not None:
            m = rescale_matrix(m, P_final[i])
        mk = markets(m)
        recs.append(dict(model=name, season=r.season, date=r.date, home=r.home, away=r.away,
                         hg=r.hg, ag=r.ag, lh=lh[i], la=la[i],
                         p_home=mk["p_home"], p_draw=mk["p_draw"], p_away=mk["p_away"],
                         p_over25=mk["p_over25"], p_btts=mk["p_btts"],
                         top_score=mk["top_scores"][0][0]))
    return pd.DataFrame(recs)


def build_model_table(matches=None):
    matches = load_matches() if matches is None else matches
    print("Bağlam verisi (seyahat, hava durumu, kupa maçları)...")
    ctx = build_context(matches)
    print("Değişkenler hesaplanıyor...")
    feats = build_features(matches, load_other_dates(), ctx)
    print("Dixon-Coles walk-forward (birkaç dakika sürebilir)...")
    dc = dixon_coles_walkforward(matches, feats)
    data = feats.merge(dc, on=["date", "home", "away"], how="inner")
    data.to_csv(ROOT / "data" / "model_table.csv", index=False)
    return data


def main():
    REPORTS.mkdir(exist_ok=True)
    data = build_model_table()
    cols = feature_columns(data)
    print(f"{len(cols)} değişken kullanılıyor")

    preds, oof = [], []  # oof: kaynak olasılıkları (birleştirme eğitimi için)
    glm_coefs = None
    for season in STACK_ONLY_SEASONS + TEST_SEASONS:
        test = data[(data.season == season) & data.played]
        train = data[(data.season < season) & data.played]
        if test.empty or len(train) < 200:
            continue
        b = base_predictions(train, test, cols)
        y = outcome(test.hg, test.ag)
        o = test[["season", "date", "home", "away", "hg", "ag"]].copy()
        for k, P in b["P"].items():
            o[[f"{k}_h", f"{k}_d", f"{k}_a"]] = P
        flh_, fla_ = final_lambdas(b)
        o[["final_h", "final_d", "final_a"]] = probs_from_lambdas(flh_, fla_, b["rho"])
        o[["mkt_h", "mkt_d", "mkt_a"]] = (test[["odds_ph", "odds_pd", "odds_pa"]].values
                                          if "odds_ph" in test else np.nan)
        oof.append(o)
        if season in STACK_ONLY_SEASONS:
            continue

        # Birleştirme: yalnızca önceki sezonların tahminleriyle eğit
        prior = pd.concat(oof[:-1])
        Pp = {k: prior[[f"{k}_h", f"{k}_d", f"{k}_a"]].values for k in b["P"]}
        params = fit_stack(Pp, outcome(prior.hg, prior.ag))
        P_stack = apply_stack(b["P"], params)
        print(f"{season}: ağırlıklar " + ", ".join(
            f"{s}={w:.2f}" for s, w in zip(params["sources"], params["weights"]))
              + f" | sıcaklık={params['temp']:.2f} beraberlik={params['draw_shift']:+.2f}")

        base_lh = np.full(len(test), train.hg.mean())
        base_la = np.full(len(test), train.ag.mean())
        preds.append(final_rows(test, base_lh, base_la, np.zeros(len(test)), None, "Taban (lig ortalaması)"))
        # Karşılaştırma: bahis piyasası (oranı olan maçlarda oran, olmayanlarda seçilen model)
        if "odds_ph" in test.columns and test.odds_ph.notna().any():
            flh0, fla0 = final_lambdas(b)
            mk_odds = test[["odds_ph", "odds_pd", "odds_pa"]].values
            fallback = probs_from_lambdas(flh0, fla0, b["rho"])
            P_mkt = np.where(np.isnan(mk_odds).any(1, keepdims=True), fallback, mk_odds)
            preds.append(final_rows(test, flh0, fla0, b["rho"], P_mkt, "Bahis piyasası (referans)"))
        names = {"dc": "Dixon-Coles", "glm": "Poisson GLM", "lgb": "LightGBM (gol)"}
        for k, (lh, la) in b["lam"].items():
            preds.append(final_rows(test, lh, la, b["rho"], None, names[k]))
        preds.append(final_rows(test, b["lh"], b["la"], b["rho"], b["P"]["lr"], "Lojistik (1X2)"))
        preds.append(final_rows(test, b["lh"], b["la"], b["rho"], b["P"]["lgbc"], "LightGBM (1X2)"))
        preds.append(final_rows(test, b["lh"], b["la"], b["rho"], None, "Topluluk (eşit ağırlık)"))
        preds.append(final_rows(test, b["lh"], b["la"], b["rho"], P_stack, "Stacking (öğrenilmiş ağırlık)"))
        flh, fla = final_lambdas(b)
        preds.append(final_rows(test, flh, fla, b["rho"], None, FINAL_NAME))
        # Kalibrasyon: önceki sezonlarda öğrenilen sıcaklık + beraberlik ayarı
        cal = fit_calibration(prior)
        P_cal = apply_calibration(probs_from_lambdas(flh, fla, b["rho"]), cal)
        print(f"{season}: kalibrasyon sıcaklık={cal['temp']:.2f} beraberlik={cal['draw_shift']:+.2f}")
        preds.append(final_rows(test, flh, fla, b["rho"], P_cal, CAL_NAME))
        # NİHAİ: oranı olan maçlarda piyasa + model (ağırlık önceki sezonlardan), yoksa kalibre model
        blend = fit_blend(prior)
        if blend and "odds_ph" in test:
            P_fin = apply_blend(probs_from_lambdas(flh, fla, b["rho"]),
                                test[["odds_ph", "odds_pd", "odds_pa"]].values, blend, P_cal)
            print(f"{season}: piyasa ağırlığı={blend['weights'][blend['sources'].index('lgb')]:.2f}")
            preds.append(final_rows(test, flh, fla, b["rho"], P_fin, BLEND_NAME))

        if season == "2024-25":
            gc = dense_columns(train, cols)
            mh = make_glm().fit(train[gc], train.hg.astype(float))
            ma = make_glm().fit(train[gc], train.ag.astype(float))
            glm_coefs = pd.DataFrame({"degisken": gc, "ev_golu_etkisi": mh[-1].coef_,
                                      "dep_golu_etkisi": ma[-1].coef_})

    # Tahmin için: tüm geçmiş tahminlerle birleştirme ağırlıkları
    allo = pd.concat(oof)
    Pall = {k: allo[[f"{k}_h", f"{k}_d", f"{k}_a"]].values for k in ["dc", "glm", "lgb", "lr", "lgbc"]}
    final_params = fit_stack(Pall, outcome(allo.hg, allo.ag))
    (REPORTS / "stack_params.json").write_text(json.dumps(final_params, indent=2), encoding="utf-8")
    (REPORTS / "calibration.json").write_text(json.dumps(fit_calibration(allo), indent=2), encoding="utf-8")
    allo.to_csv(REPORTS / "oof_probs.csv", index=False)
    fb = fit_blend(allo)
    if fb:
        (REPORTS / "blend.json").write_text(json.dumps(fb, indent=2), encoding="utf-8")  # sürpriz modeli bu tahminlerle eğitilir

    P = pd.concat(preds, ignore_index=True)
    P.to_csv(REPORTS / "backtest_predictions.csv", index=False)
    summary = []
    for (model, season), g in P.groupby(["model", "season"], sort=False):
        summary.append(dict(model=model, season=season, **evaluate(g)))
    for model, g in P.groupby("model", sort=False):
        summary.append(dict(model=model, season="TOPLAM", **evaluate(g)))
    S = pd.DataFrame(summary)
    S.to_csv(REPORTS / "backtest_summary.csv", index=False)
    if glm_coefs is not None:
        glm_coefs["onem"] = glm_coefs[["ev_golu_etkisi", "dep_golu_etkisi"]].abs().max(1)
        glm_coefs.sort_values("onem", ascending=False).to_csv(REPORTS / "feature_effects.csv", index=False)

    pd.set_option("display.width", 200)
    show = ["model", "season", "n", "logloss_1x2", "rps", "acc_1x2", "acc_o25", "logloss_btts", "exact_score_hit"]
    print(S[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
