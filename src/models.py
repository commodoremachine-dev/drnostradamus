"""Skor modelleri ve skor matrisinden türetilen olasılıklar.

1) DixonColes: takım hücum/savunma gücü + ev avantajı, zaman ağırlıklı (eski maçlar
   daha az etkili) ve düşük skorlar (0-0, 1-0, 0-1, 1-1) için rho düzeltmesi.
2) Özellik tabanlı modeller (features.py çıktısı + Dixon-Coles gücü):
   ev golü ve deplasman golü için ayrı Poisson regresyonu / LightGBM.
Hepsi sonunda beklenen gol (lambda) üretir; skor matrisi bunlardan kurulur.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 10


def valid_rho(lh, la, rho, eps=1e-10):
    """Tüm dört düşük skor hücresini pozitif tutan rho izdüşümü."""
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    if (not np.isfinite(lh).all() or not np.isfinite(la).all()
            or (lh <= 0).any() or (la <= 0).any() or not np.isfinite(rho)):
        raise ValueError("Gol beklentileri pozitif ve sonlu, rho sonlu olmalı.")
    lower = max(float(np.max(-1 / lh)), float(np.max(-1 / la))) + eps
    upper = min(1.0, float(np.min(1 / (lh * la)))) - eps
    return float(np.clip(rho, lower, upper))


# ---------------------------------------------------------------- skor matrisi
def tau_matrix(lh, la, rho, n=MAX_GOALS):
    if n < 2:
        raise ValueError("Skor matrisi boyutu en az 2 olmalı.")
    rho = valid_rho(lh, la, rho)
    t = np.ones((n, n))
    t[0, 0] = 1 - lh * la * rho
    t[0, 1] = 1 + lh * rho
    t[1, 0] = 1 + la * rho
    t[1, 1] = 1 - rho
    return t


def score_matrix(lh, la, rho=0.0, n=MAX_GOALS):
    rho = valid_rho(lh, la, rho)
    if n < 2:
        raise ValueError("Skor matrisi boyutu en az 2 olmalı.")
    # Yüksek gol beklentilerinde 0..9'a kesmek olasılıkları çarpıtmasın.
    n = max(n, int(poisson.ppf(1 - 1e-10, max(lh, la))) + 1)
    m = np.outer(poisson.pmf(np.arange(n), lh), poisson.pmf(np.arange(n), la))
    if rho:
        m = m * tau_matrix(lh, la, rho, n)
    return m / m.sum()


def markets(m: np.ndarray) -> dict:
    n = m.shape[0]
    i, j = np.indices(m.shape)
    top = np.dstack(np.unravel_index(np.argsort(-m, axis=None)[:5], m.shape))[0]
    return {
        "p_home": float(np.tril(m, -1).sum()),
        "p_draw": float(np.trace(m)),
        "p_away": float(np.triu(m, 1).sum()),
        "p_over15": float(m[(i + j) > 1].sum()),
        "p_over25": float(m[(i + j) > 2].sum()),
        "p_over35": float(m[(i + j) > 3].sum()),
        "p_btts": float(m[1:, 1:].sum()),
        "xg_home": float((m.sum(1) * np.arange(n)).sum()),
        "xg_away": float((m.sum(0) * np.arange(n)).sum()),
        "top_scores": [(f"{a}-{b}", float(m[a, b])) for a, b in top],
    }


# ---------------------------------------------------------------- Dixon-Coles
class DixonColes:
    def __init__(self, xi=0.0019, l2=0.5):
        self.xi = xi      # günlük zaman sönümü (0.0019 ≈ 1 yıl yarı ömür)
        self.l2 = l2      # az maçı olan takımları ortalamaya çeker
        self.teams = []

    def fit(self, df: pd.DataFrame, ref_date):
        if self.xi < 0 or self.l2 < 0:
            raise ValueError("Zaman sönümü ve düzenlileştirme negatif olamaz.")
        d = df[df.played & (df.date < pd.Timestamp(ref_date))].dropna(subset=["hg", "ag"])
        age = (pd.Timestamp(ref_date) - d.date).dt.days.values
        keep = age < 365 * 4
        d, age = d[keep], age[keep]
        if d.empty:
            raise ValueError("Referans tarihinden önce oynanmış eğitim maçı yok.")
        goals = d[["hg", "ag"]].to_numpy(dtype=float)
        if not np.isfinite(goals).all() or (goals < 0).any() or (goals != np.floor(goals)).any():
            raise ValueError("Goller negatif olmayan tam sayılar olmalı.")
        w = np.exp(-self.xi * age)
        self.teams = sorted(set(d.home) | set(d.away))
        idx = {t: i for i, t in enumerate(self.teams)}
        h = d.home.map(idx).values
        a = d.away.map(idx).values
        hg = d.hg.values.astype(float)
        ag = d.ag.values.astype(float)
        T = len(self.teams)

        def nll(p):
            mu, home, att, dfn = p[0], p[1], p[2:2 + T], p[2 + T:]
            llh = mu + home + att[h] + dfn[a]
            lla = mu + att[a] + dfn[h]
            lh, la = np.exp(llh), np.exp(lla)
            ll = w * (hg * llh - lh + ag * lla - la)
            f = -ll.sum() + self.l2 * (att @ att + dfn @ dfn)
            rh = w * (hg - lh)
            ra = w * (ag - la)
            g = np.zeros_like(p)
            g[0] = -(rh.sum() + ra.sum())
            g[1] = -rh.sum()
            g[2:2 + T] = -(np.bincount(h, rh, T) + np.bincount(a, ra, T)) + 2 * self.l2 * att
            g[2 + T:] = -(np.bincount(a, rh, T) + np.bincount(h, ra, T)) + 2 * self.l2 * dfn
            return f, g

        p0 = np.zeros(2 + 2 * T)
        p0[0] = np.log(max(ag.mean(), 0.5))
        res = minimize(nll, p0, jac=True, method="L-BFGS-B")
        if not res.success or not np.isfinite(res.fun):
            raise RuntimeError(f"Dixon-Coles optimizasyonu yakınsamadı: {res.message}")
        p = res.x
        # Merkezleme gol beklentilerini korur; takım güçlerinin yorumu sabitlenir.
        am, dm = p[2:2 + T].mean(), p[2 + T:].mean()
        p[0] += am + dm
        p[2:2 + T] -= am
        p[2 + T:] -= dm
        self.mu, self.home = p[0], p[1]
        self.att = dict(zip(self.teams, p[2:2 + T]))
        self.dfn = dict(zip(self.teams, p[2 + T:]))

        # rho'yu ayrı (tek boyutlu) bul
        lh = np.exp(self.mu + self.home + p[2:2 + T][h] + p[2 + T:][a])
        la = np.exp(self.mu + p[2:2 + T][a] + p[2 + T:][h])

        def rho_nll(rho):
            t = np.ones_like(lh)
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            t[m00] = 1 - lh[m00] * la[m00] * rho
            t[m01] = 1 + lh[m01] * rho
            t[m10] = 1 + la[m10] * rho
            t[m11] = 1 - rho
            return -(w * np.log(np.clip(t, 1e-9, None))).sum()

        bounds = (valid_rho(lh, la, -0.2), valid_rho(lh, la, 0.2))
        self.rho = minimize_scalar(rho_nll, bounds=bounds, method="bounded").x
        self.train_count = len(d)
        self.train_end = d.date.max()
        return self

    def strength(self, team, promoted=False):
        # Hiç görülmemiş takım: yükselen takımlar için ortalamadan biraz zayıf varsay
        default_att, default_def = (-0.15, 0.15) if promoted else (0.0, 0.0)
        return self.att.get(team, default_att), self.dfn.get(team, default_def)

    def lambdas(self, home, away, home_promoted=False, away_promoted=False):
        ah, dh = self.strength(home, home_promoted)
        aa, da = self.strength(away, away_promoted)
        return (np.exp(self.mu + self.home + ah + da), np.exp(self.mu + aa + dh))


# ---------------------------------------------------------------- metrikler
def rps(probs: np.ndarray, outcome: np.ndarray) -> float:
    """Ranked Probability Score (1X2 sıralı). Düşük = iyi."""
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    obs = np.zeros_like(probs)
    obs[np.arange(len(outcome)), outcome] = 1
    cum_o = np.cumsum(obs, axis=1)[:, :2]
    return float(((cum_p - cum_o) ** 2).sum(1).mean() / 2)


def evaluate(pred: pd.DataFrame) -> dict:
    """pred: hg, ag, p_home, p_draw, p_away, p_over25, p_btts, top_score sütunları."""
    y = np.where(pred.hg > pred.ag, 0, np.where(pred.hg == pred.ag, 1, 2))
    P = np.clip(pred[["p_home", "p_draw", "p_away"]].values, 1e-9, 1)
    o25 = (pred.hg + pred.ag > 2).values
    btts = ((pred.hg > 0) & (pred.ag > 0)).values
    po = np.clip(pred.p_over25.values, 1e-9, 1 - 1e-9)
    pb = np.clip(pred.p_btts.values, 1e-9, 1 - 1e-9)
    actual_score = pred.hg.astype(int).astype(str) + "-" + pred.ag.astype(int).astype(str)
    return {
        "n": len(pred),
        "logloss_1x2": float(-np.log(P[np.arange(len(y)), y]).mean()),
        "rps": rps(P, y),
        "brier_1x2": float(((P - np.eye(3)[y]) ** 2).sum(1).mean()),
        "acc_1x2": float((P.argmax(1) == y).mean()),
        "logloss_o25": float(-(o25 * np.log(po) + (~o25) * np.log(1 - po)).mean()),
        "acc_o25": float(((po > 0.5) == o25).mean()),
        "logloss_btts": float(-(btts * np.log(pb) + (~btts) * np.log(1 - pb)).mean()),
        "exact_score_hit": float((pred.top_score.values == actual_score.values).mean()),
    }
