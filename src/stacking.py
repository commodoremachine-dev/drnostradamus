"""Modelleri birleştirme (stacking) ve kalibrasyon.

Kaynaklar (her biri 1/X/2 olasılığı verir):
  dc   : Dixon-Coles skor matrisi
  glm  : Poisson GLM skor matrisi
  lgb  : LightGBM Poisson skor matrisi
  lr   : doğrudan 1/X/2 lojistik regresyon
  lgbc : doğrudan 1/X/2 LightGBM sınıflandırıcı
Ağırlıklar ve beraberlik/sıcaklık kalibrasyonu yalnızca GEÇMİŞ sezon tahminlerinde
log loss en aza indirilerek öğrenilir.
"""
import numpy as np
from scipy.optimize import minimize

SOURCES = ["dc", "glm", "lgb", "lr", "lgbc"]


def _combine(P_list, w, temp, draw_shift):
    # Log-olasılıkların ağırlıklı ortalaması (geometrik havuzlama) + kalibrasyon
    logp = sum(wi * np.log(np.clip(P, 1e-9, 1)) for wi, P in zip(w, P_list))
    logp = logp / temp
    logp[:, 1] += draw_shift
    e = np.exp(logp - logp.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _unpack(x, k):
    w = np.exp(x[:k]) / np.exp(x[:k]).sum()
    return w, np.exp(x[k]), x[k + 1]


def fit_stack(P_dict: dict, y: np.ndarray, l2=0.02) -> dict:
    srcs = [s for s in SOURCES if s in P_dict]
    P_list = [P_dict[s] for s in srcs]
    k = len(srcs)

    def loss(x):
        w, t, d = _unpack(x, k)
        P = _combine(P_list, w, t, d)
        return -np.log(P[np.arange(len(y)), y]).mean() + l2 * (x[:k] ** 2).sum() / k

    res = minimize(loss, np.zeros(k + 2), method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
    w, t, d = _unpack(res.x, k)
    return {"sources": srcs, "weights": w.tolist(), "temp": float(t), "draw_shift": float(d)}


def apply_stack(P_dict: dict, params: dict) -> np.ndarray:
    return _combine([P_dict[s] for s in params["sources"]], params["weights"],
                    params["temp"], params["draw_shift"])


def rescale_matrix(m: np.ndarray, p_target) -> np.ndarray:
    """Skor matrisini 1/X/2 bölgeleri hedef olasılıklara uyacak şekilde ölçekler."""
    m = m.copy()
    i, j = np.indices(m.shape)
    for mask, p in ((i > j, p_target[0]), (i == j, p_target[1]), (i < j, p_target[2])):
        s = m[mask].sum()
        if s > 0:
            m[mask] *= p / s
    return m / m.sum()
