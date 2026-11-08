"""Bir tahminin gerekçelerini sade cümlelerle üretir (arayüzdeki "Neden?" bölümü)."""
import numpy as np
import pandas as pd


def _v(r, k):
    v = getattr(r, k, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def explain(r, fin: dict, mod: dict) -> list[tuple[str, str]]:
    """(taraf, metin) listesi. taraf: 'ev', 'dep' ya da 'nötr'."""
    H, A = r.home, r.away
    out = []
    out.append(("ev", f"İç saha avantajı: Süper Lig'de ev sahibi maçların ~%44'ünü, deplasman ~%29'unu kazanıyor."))

    e = _v(r, "elo_home") - _v(r, "elo_away")
    if not np.isnan(e):
        if abs(e) < 25:
            out.append(("nötr", f"Güç (Elo) neredeyse eşit: {_v(r, 'elo_home'):.0f} - {_v(r, 'elo_away'):.0f}."))
        else:
            out.append(("ev" if e > 0 else "dep",
                        f"Güç (Elo): {H if e > 0 else A} {abs(e):.0f} puan önde ({_v(r, 'elo_home'):.0f} - {_v(r, 'elo_away'):.0f})."))

    pi = _v(r, "pi_exp_gd")
    if not np.isnan(pi) and abs(pi) >= 0.2:
        out.append(("ev" if pi > 0 else "dep",
                    f"Pi-rating (iç saha / deplasman gücü): {'ev sahibi' if pi > 0 else 'deplasman'} lehine "
                    f"~{abs(pi):.1f} gol beklenti farkı."))

    f_h, f_a = _v(r, "home_ppg5"), _v(r, "away_ppg5")
    if not np.isnan(f_h) and not np.isnan(f_a) and abs(f_h - f_a) >= 0.6:
        better = H if f_h > f_a else A
        out.append(("ev" if f_h > f_a else "dep",
                    f"Form: son 5 maçta puan ortalaması {H} {f_h:.1f}, {A} {f_a:.1f} — {better} daha formda."))

    for side, t in (("home", H), ("away", A)):
        if _v(r, f"{side}_promoted") == 1:
            out.append(("dep" if side == "home" else "ev", f"{t} bu sezon yükseldi; model yeni takımlara temkinli yaklaşır."))

    ga = _v(r, "home_ga_venue_ewm")
    if not np.isnan(ga) and ga >= 1.8:
        out.append(("dep", f"{H} evinde maç başına ~{ga:.1f} gol yiyor (zayıf iç saha savunması)."))
    gf = _v(r, "away_gf_venue_ewm")
    if not np.isnan(gf) and gf >= 1.6:
        out.append(("dep", f"{A} deplasmanda maç başına ~{gf:.1f} gol atıyor."))

    h2h, n = _v(r, "h2h_gd_home"), _v(r, "h2h_n")
    if not np.isnan(h2h) and n >= 3 and abs(h2h) >= 0.5:
        out.append(("ev" if h2h > 0 else "dep", f"İkili rekabet: son {int(n)} maçta {H if h2h > 0 else A} "
                                                f"ortalama {abs(h2h):.1f} gol farkla üstün."))

    for side, t in (("home", H), ("away", A)):
        oc = _v(r, f"{side}_other_comp_14d")
        rd = _v(r, f"{side}_rest_days")
        if oc >= 1 or (not np.isnan(rd) and rd <= 3):
            out.append(("dep" if side == "home" else "ev",
                        f"{t} yorgun olabilir: {('son 14 günde Avrupa maçı, ' if oc >= 1 else '')}"
                        f"{'' if np.isnan(rd) else f'{rd:.0f} gün dinlenme'}."))
        rm = _v(r, f"{side}_regulars_missing")
        if rm >= 3:
            out.append(("dep" if side == "home" else "ev", f"{t} ilk 11'inde {int(rm)} asıl oyuncu yok."))

    km = _v(r, "away_travel_km")
    if km >= 800:
        out.append(("ev", f"{A} için uzun deplasman yolu (~{km:.0f} km)."))
    rain = _v(r, "w_rain_mm")
    if rain >= 10:
        out.append(("nötr", f"Maç günü şiddetli yağış bekleniyor ({rain:.0f} mm) — sonuç daha rastgele olabilir."))

    ph = _v(r, "odds_ph")
    if not np.isnan(ph):
        out.append(("ev" if ph > _v(r, "odds_pa") else "dep",
                    f"Bahis piyasası: 1 %{100 * ph:.0f} / X %{100 * _v(r, 'odds_pd'):.0f} / 2 %{100 * _v(r, 'odds_pa'):.0f}."))
        out.append(("nötr", f"Model tek başına: 1 %{100 * mod['p_home']:.0f} / X %{100 * mod['p_draw']:.0f} / "
                            f"2 %{100 * mod['p_away']:.0f}; nihai tahmin piyasa ile birleştirilip kalibre edildi."))
    else:
        out.append(("nötr", "Bu maç için piyasa oranı yok; tahmin yalnızca modele dayanıyor."))

    ts = fin["top_scores"][0][0]
    i, j = map(int, ts.split("-"))
    res_top = "1" if i > j else "X" if i == j else "2"
    pick = ["1", "X", "2"][int(np.argmax([fin["p_home"], fin["p_draw"], fin["p_away"]]))]
    if res_top != pick:
        out.append(("nötr", f"En olası tek skor {ts}, ama {pick} sonucu birçok skorun toplamıyla daha olası."))
    return out
