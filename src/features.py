"""Her maç için, yalnızca o maçtan ÖNCEKİ verilerle değişken (özellik) hesaplar.

Kronolojik tek geçiş yapılır; böylece gelecekten bilgi sızmaz.
Hesaplanan değişken grupları:
  - Elo gücü (ev avantajı, gol farkı çarpanı, sezon başı ortalamaya dönüş)
  - Form: son 5/10 maç puan ortalaması, seri (yenilmezlik / galibiyetsizlik)
  - Gol eğilimleri: atılan/yenen gol (üstel ağırlıklı), iç saha / deplasman ayrı
  - Oranlar: gol yememe, gol atamama, 2.5 üst, karşılıklı gol
  - İlk yarı verisi: ilk yarı gol ortalaması, öne geçip kaybetme, geriden gelme
  - Beklenti sapması: Elo'nun beklediği puana göre fazla/eksik performans
  - Takvim: dinlenme günü, son 21 günde oynanan maç sayısı
  - Sezon durumu: puan ortalaması, averaj, lig sırası, sezon ilerlemesi
  - Bağlam: derbi, yeni yükselen takım, ikili rekabet geçmişi
  - (Varsa) şut, isabetli şut, korner, kart istatistikleri
"""
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BIG = {"Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor"}
EXTRA_STATS = [("hs", "as_"), ("hst", "ast"), ("hc", "ac"), ("hy", "ay"), ("hr", "ar"),
               ("hxg", "axg"), ("hposs", "aposs"),
               ("hsib", "asib"),          # ceza sahası içi şut
               ("hpass", "apass"),        # pas isabet %
               ("hduel", "aduel"),        # ikili mücadele kazanma %
               ("hdrb", "adrb"),          # başarılı çalım %
               ("hkp", "akp"),            # kilit pas
               ("htkl", "atkl"),          # top kapma + araya girme
               ("hrating", "arating"),    # oyuncu reytingi ort. (dakika ağırlıklı)
               ("hpen", "apen")]          # kazanılan penaltı

ELO_K, ELO_HOME, ELO_START, ELO_PROMOTED = 22.0, 65.0, 1500.0, 1430.0
ELO_CARRY = 0.75  # sezon başında Elo farkının korunan oranı
EWM_ALPHA = 0.18  # yaklaşık son 10 maça ağırlık


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "matches.csv", parse_dates=["date"])
    return df.sort_values(["date", "home"]).reset_index(drop=True)


class TeamState:
    def __init__(self):
        self.hist = deque(maxlen=40)  # (date, is_home, gf, ga, htgf, htga, pts, exp_pts, extra)
        self.ewm = {}

    def upd_ewm(self, key, val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return
        old = self.ewm.get(key)
        self.ewm[key] = val if old is None else (1 - EWM_ALPHA) * old + EWM_ALPHA * val


def _elo_expect(r_home, r_away):
    return 1.0 / (1.0 + 10 ** ((r_away - (r_home + ELO_HOME)) / 400.0))


def _team_feats(st: TeamState, date, prefix, venue_home: bool) -> dict:
    h = list(st.hist)
    f = {}
    last5, last10 = h[-5:], h[-10:]
    f["n_games"] = len(h)
    f["ppg5"] = np.mean([x[6] for x in last5]) if last5 else np.nan
    f["ppg10"] = np.mean([x[6] for x in last10]) if last10 else np.nan
    f["gf_ewm"] = st.ewm.get("gf", np.nan)
    f["ga_ewm"] = st.ewm.get("ga", np.nan)
    vk = "home" if venue_home else "away"
    f["gf_venue_ewm"] = st.ewm.get(f"gf_{vk}", np.nan)
    f["ga_venue_ewm"] = st.ewm.get(f"ga_{vk}", np.nan)
    f["cs_rate10"] = np.mean([x[3] == 0 for x in last10]) if last10 else np.nan
    f["fts_rate10"] = np.mean([x[2] == 0 for x in last10]) if last10 else np.nan
    f["o25_rate10"] = np.mean([x[2] + x[3] > 2 for x in last10]) if last10 else np.nan
    f["btts_rate10"] = np.mean([x[2] > 0 and x[3] > 0 for x in last10]) if last10 else np.nan
    f["ht_gf_ewm"] = st.ewm.get("htgf", np.nan)
    f["ht_ga_ewm"] = st.ewm.get("htga", np.nan)
    ht = [x for x in h[-20:] if x[4] is not None and not pd.isna(x[4])]
    led = [x for x in ht if x[4] > x[5]]
    trail = [x for x in ht if x[4] < x[5]]
    f["lead_lost_rate"] = np.mean([x[6] < 3 for x in led]) if len(led) >= 3 else np.nan
    f["comeback_rate"] = np.mean([x[6] > 0 for x in trail]) if len(trail) >= 3 else np.nan
    f["over_perf10"] = np.mean([x[6] - x[7] for x in last10]) if last10 else np.nan
    unb = 0
    for x in reversed(h):
        if x[6] == 0:
            break
        unb += 1
    winless = 0
    for x in reversed(h):
        if x[6] == 3:
            break
        winless += 1
    f["unbeaten_streak"], f["winless_streak"] = unb, winless
    f["rest_days"] = min((date - h[-1][0]).days, 30) if h else np.nan
    f["games_21d"] = sum((date - x[0]).days <= 21 for x in h)
    for hk, _ in EXTRA_STATS:
        for side in ("for", "ag"):
            k = f"{hk}_{side}"
            if k in st.ewm:
                f[f"{hk.rstrip('_')}_{side}_ewm"] = st.ewm[k]
    return {f"{prefix}_{k}": v for k, v in f.items()}


PI_LAMBDA, PI_GAMMA, PI_C = 0.035, 0.7, 3.0  # Constantinou & Fenton (2013)


def _pi_goal(r):
    """Pi-rating değerini beklenen gol farkına çevirir."""
    return np.sign(r) * (10 ** (abs(r) / PI_C) - 1)


def _xpts(xg_for, xg_against, n=10):
    """xG'den beklenen puan: iki bağımsız Poisson ile galibiyet/beraberlik olasılığı."""
    from scipy.stats import poisson
    p1 = poisson.pmf(np.arange(n), xg_for)
    p2 = poisson.pmf(np.arange(n), xg_against)
    m = np.outer(p1, p2)
    return 3 * np.tril(m, -1).sum() + np.trace(m)


def build_features(df: pd.DataFrame, other_dates: dict | None = None,
                   context: pd.DataFrame | None = None) -> pd.DataFrame:
    """other_dates: takım -> diğer turnuva (kupa/Avrupa) maç tarihleri listesi.
    context: date/home/away anahtarlı hava durumu ve seyahat tablosu."""
    other_dates = other_dates or {}
    elo = defaultdict(lambda: ELO_START)
    pi = defaultdict(lambda: [0.0, 0.0])  # takım -> [iç saha puanı, deplasman puanı]
    ref_hist = defaultdict(list)  # hakem -> [(kart, penaltı, ev kazandı mı)]
    xi_hist = defaultdict(lambda: deque(maxlen=10))  # takım -> son ilk 11'ler (oyuncu id kümeleri)
    has_xi = "h_xi" in df.columns
    states: dict[str, TeamState] = defaultdict(TeamState)
    h2h = defaultdict(list)  # (a,b) sıralı çift -> [(date, home, gd_home)]
    season_teams: dict[str, set] = df.groupby("season").apply(
        lambda x: set(x.home) | set(x.away), include_groups=False).to_dict()
    seasons = sorted(season_teams)
    table = {}  # sezon içi puan tablosu
    prev_final = {}
    cur_season = None
    has_extra = [c for pair in EXTRA_STATS for c in pair if c in df.columns]
    has_missing = "h_missing" in df.columns

    out = []
    for r in df.itertuples(index=False):
        if r.season != cur_season:
            # Sezon başı: Elo'yu ortalamaya biraz çek, yükselen takımları belirle
            prev = seasons[seasons.index(r.season) - 1] if seasons.index(r.season) > 0 else None
            prev_teams = season_teams.get(prev, set())
            for t in list(elo):
                elo[t] = ELO_START + (elo[t] - ELO_START) * ELO_CARRY
            for t in season_teams[r.season]:
                if prev is not None and t not in prev_teams and t not in elo:
                    elo[t] = ELO_PROMOTED
            promoted = {t for t in season_teams[r.season] if prev is not None and t not in prev_teams}
            # Geçen sezonun final tablosu (sezon başında form verisi boşken yol gösterir)
            if table:
                order = sorted(table, key=lambda t: (-table[t]["p"], -table[t]["gd"]))
                prev_final = {t: (table[t]["p"] / table[t]["n"] if table[t]["n"] else np.nan, i + 1)
                              for i, t in enumerate(order)}
            else:
                prev_final = {}
            table = {t: dict(p=0, gd=0, n=0, xp=0.0, nx=0) for t in season_teams[r.season]}
            cur_season = r.season
            n_season_matches = len(df[df.season == r.season])

        hs_, as_ = states[r.home], states[r.away]
        e_h = _elo_expect(elo[r.home], elo[r.away])
        rank = {t: i + 1 for i, t in enumerate(sorted(table, key=lambda t: (-table[t]["p"], -table[t]["gd"])))}
        row = dict(
            season=r.season, date=r.date, round=r.round, home=r.home, away=r.away,
            hg=r.hg, ag=r.ag, played=r.played,
            elo_home=elo[r.home], elo_away=elo[r.away], elo_diff=elo[r.home] - elo[r.away],
            elo_exp_home=e_h,
            is_derby=int(r.home in BIG and r.away in BIG),
            home_big=int(r.home in BIG), away_big=int(r.away in BIG),
            home_promoted=int(r.home in promoted), away_promoted=int(r.away in promoted),
            season_progress=sum(t["n"] for t in table.values()) / 2 / max(n_season_matches, 1),
            month=r.date.month,
        )
        for side, t in (("home", r.home), ("away", r.away)):
            tb = table[t]
            row[f"{side}_season_ppg"] = tb["p"] / tb["n"] if tb["n"] else np.nan
            row[f"{side}_season_gdpg"] = tb["gd"] / tb["n"] if tb["n"] else np.nan
            row[f"{side}_rank"] = rank[t] if tb["n"] else np.nan
            # xPTS: sezon boyunca xG'ye göre hak edilen puan ortalaması ve gerçek puanla farkı
            if tb["nx"]:
                row[f"{side}_season_xppg"] = tb["xp"] / tb["nx"]
                row[f"{side}_season_pts_minus_xpts"] = (tb["p"] - tb["xp"]) / tb["nx"] if tb["nx"] == tb["n"] else np.nan
        for side, t in (("home", r.home), ("away", r.away)):
            pf = prev_final.get(t)
            row[f"{side}_prev_ppg"] = pf[0] if pf else np.nan
            row[f"{side}_prev_rank"] = pf[1] if pf else np.nan
            row[f"{side}_season_games"] = table[t]["n"]
        # Pi-rating: ev sahibinin iç saha, deplasmanın deplasman puanı
        pi_h, pi_a = pi[r.home][0], pi[r.away][1]
        row["pi_home"], row["pi_away"] = pi_h, pi_a
        row["pi_exp_gd"] = _pi_goal(pi_h) - _pi_goal(pi_a)
        row.update(_team_feats(hs_, r.date, "home", True))
        row.update(_team_feats(as_, r.date, "away", False))
        # Takıma özel iç saha avantajı: iç saha puanı - deplasman puanı (ağırlıklı)
        for side, st in (("home", hs_), ("away", as_)):
            ph_, pa_ = st.ewm.get("pts_home"), st.ewm.get("pts_away")
            row[f"{side}_home_pts_ewm"] = ph_ if ph_ is not None else np.nan
            row[f"{side}_away_pts_ewm"] = pa_ if pa_ is not None else np.nan
            row[f"{side}_home_adv"] = ph_ - pa_ if ph_ is not None and pa_ is not None else np.nan
            if "xpts" in st.ewm:
                row[f"{side}_xpts_ewm"] = st.ewm["xpts"]
        # Kupa / Avrupa maçları dahil yorgunluk
        if other_dates:
            for side, t in (("home", r.home), ("away", r.away)):
                od = [d for d in other_dates.get(t, ()) if d < r.date]
                last_league = row[f"{side}_rest_days"]
                if od:
                    rest_all = min((r.date - max(od)).days, 30)
                    row[f"{side}_rest_days"] = rest_all if pd.isna(last_league) else min(last_league, rest_all)
                row[f"{side}_games_21d"] += sum((r.date - d).days <= 21 for d in od)
                row[f"{side}_other_comp_14d"] = sum((r.date - d).days <= 14 for d in od)
        # Hakem: geçmiş maçlarındaki kart, penaltı ve ev sahibi galibiyet oranı
        ref = getattr(r, "referee", None)
        if isinstance(ref, str) and ref:
            rh = ref_hist[ref][-30:]
            row["ref_n"] = len(rh)
            if len(rh) >= 3:
                cards = [x[0] for x in rh if not pd.isna(x[0])]
                pens = [x[1] for x in rh if not pd.isna(x[1])]
                row["ref_cards_pg"] = np.mean(cards) if cards else np.nan
                row["ref_pens_pg"] = np.mean(pens) if pens else np.nan
                row["ref_home_win_rate"] = np.mean([x[2] for x in rh])
        # Hava durumu ve seyahat
        if context is not None:
            c = context.get((r.date, r.home, r.away))
            if c:
                row.update(c)
        key = tuple(sorted((r.home, r.away)))
        past = h2h[key][-6:]
        row["h2h_n"] = len(past)
        row["h2h_gd_home"] = (np.mean([g if hm == r.home else -g for _, hm, g in past])
                              if past else np.nan)
        # Maç öncesi bilinen eksik oyuncu sayısı (API sakatlık/ceza listesi)
        if has_missing:
            row["home_missing"] = r.h_missing
            row["away_missing"] = r.a_missing
            row["diff_missing"] = (r.h_missing - r.a_missing
                                   if not (pd.isna(r.h_missing) or pd.isna(r.a_missing)) else np.nan)
        # xG performansı: atılan gol - xG (şans/bitiricilik), son maçlar ağırlıklı
        for side, st in (("home", hs_), ("away", as_)):
            if "xg_luck" in st.ewm:
                row[f"{side}_xg_luck_ewm"] = st.ewm["xg_luck"]
        # İlk 11: rotasyon ve kilit oyuncu eksikliği (ilk 11 maçtan ~1 saat önce açıklanır)
        if has_xi:
            for side, t, xs in (("home", r.home, r.h_xi), ("away", r.away, r.a_xi)):
                xi = set(str(xs).split("|")) if isinstance(xs, str) else None
                hist = xi_hist[t]
                if xi and len(hist) >= 3:
                    starts = {}
                    for past in hist:
                        for pid in past:
                            starts[pid] = starts.get(pid, 0) + 1
                    share = {pid: c / len(hist) for pid, c in starts.items()}
                    regulars = {pid for pid, v in share.items() if v >= 0.7}
                    row[f"{side}_xi_continuity"] = len(xi & hist[-1]) / 11
                    row[f"{side}_xi_regulars"] = float(np.mean([share.get(pid, 0.0) for pid in xi]))
                    row[f"{side}_regulars_missing"] = len(regulars - xi)
                    row[f"{side}_xi_new_faces"] = sum(1 for pid in xi if pid not in starts)
            for k in ("xi_regulars", "regulars_missing"):
                if f"home_{k}" in row and f"away_{k}" in row:
                    row[f"diff_{k}"] = row[f"home_{k}"] - row[f"away_{k}"]
        # Maç öncesi bahis oranlarından çıkarılan (marjı temizlenmiş) olasılıklar
        if "odds_ph" in df.columns:
            ph_, pd_, pa_ = r.odds_ph, r.odds_pd, r.odds_pa
            row["odds_ph"], row["odds_pd"], row["odds_pa"] = ph_, pd_, pa_
            row["odds_log_ratio"] = (np.log(ph_ / pa_) if not (pd.isna(ph_) or pd.isna(pa_)) and pa_ > 0
                                     else np.nan)
            row["has_odds"] = int(not pd.isna(ph_))
        if "odds_pover25" in df.columns:
            row["odds_pover25"] = r.odds_pover25
        # Oyuncu bazlı API değişkenleri (kilit oyuncu eksikliği) maç öncesi bilinir
        for c in ("h_missing_min_share", "a_missing_min_share", "h_missing_ga_share", "a_missing_ga_share"):
            if c in df.columns:
                v = getattr(r, c)
                row[("home_" if c[0] == "h" else "away_") + c[2:]] = v
        # Ev - deplasman farkları (modeller için kullanışlı)
        for k in ["ppg5", "ppg10", "gf_ewm", "ga_ewm", "over_perf10", "season_ppg", "rest_days",
                  "xpts_ewm", "home_adv"]:
            row[f"diff_{k}"] = row.get(f"home_{k}", np.nan) - row.get(f"away_{k}", np.nan)
        out.append(row)

        if not r.played or pd.isna(r.hg):
            continue
        # --- Maç sonrası durum güncellemesi ---
        hg, ag = int(r.hg), int(r.ag)
        hthg = None if pd.isna(r.hthg) else int(r.hthg)
        htag = None if pd.isna(r.htag) else int(r.htag)
        ph = 3 if hg > ag else 1 if hg == ag else 0
        pa = 3 if ag > hg else 1 if hg == ag else 0
        exp_ph = 3 * e_h - 0.5  # kaba beklenen puan (beraberlik payı ile)
        exp_pa = 3 * (1 - e_h) - 0.5
        hs_.hist.append((r.date, True, hg, ag, hthg, htag, ph, exp_ph))
        as_.hist.append((r.date, False, ag, hg, htag, hthg, pa, exp_pa))
        for st, gf, ga, hgf, hga, vk in ((hs_, hg, ag, hthg, htag, "home"), (as_, ag, hg, htag, hthg, "away")):
            st.upd_ewm("gf", gf)
            st.upd_ewm("ga", ga)
            st.upd_ewm(f"gf_{vk}", gf)
            st.upd_ewm(f"ga_{vk}", ga)
            st.upd_ewm("htgf", hgf)
            st.upd_ewm("htga", hga)
        for hk, ak in EXTRA_STATS:
            if hk in has_extra:
                hv, av = getattr(r, hk), getattr(r, ak)
                hv = np.nan if pd.isna(hv) else float(hv)
                av = np.nan if pd.isna(av) else float(av)
                hs_.upd_ewm(f"{hk}_for", hv)
                hs_.upd_ewm(f"{hk}_ag", av)
                as_.upd_ewm(f"{hk}_for", av)
                as_.upd_ewm(f"{hk}_ag", hv)
        if has_xi:
            for t, xs in ((r.home, r.h_xi), (r.away, r.a_xi)):
                if isinstance(xs, str):
                    xi_hist[t].append(set(xs.split("|")))
        hs_.upd_ewm("pts_home", ph)
        as_.upd_ewm("pts_away", pa)
        if "hxg" in has_extra and not (pd.isna(r.hxg) or pd.isna(r.axg)):
            hxg, axg = float(r.hxg), float(r.axg)
            hs_.upd_ewm("xg_luck", hg - hxg)
            as_.upd_ewm("xg_luck", ag - axg)
            xph, xpa = _xpts(hxg, axg), _xpts(axg, hxg)
            hs_.upd_ewm("xpts", xph)
            as_.upd_ewm("xpts", xpa)
            for t, xp in ((r.home, xph), (r.away, xpa)):
                table[t]["xp"] += xp
                table[t]["nx"] += 1
        # Pi-rating güncelle
        g_hat = _pi_goal(pi[r.home][0]) - _pi_goal(pi[r.away][1])
        err = abs((hg - ag) - g_hat)
        psi = PI_C * np.log10(1 + err)
        psi_h = psi if g_hat < (hg - ag) else -psi
        old_hh, old_aa = pi[r.home][0], pi[r.away][1]
        pi[r.home][0] += psi_h * PI_LAMBDA
        pi[r.home][1] += (pi[r.home][0] - old_hh) * PI_GAMMA
        pi[r.away][1] += -psi_h * PI_LAMBDA
        pi[r.away][0] += (pi[r.away][1] - old_aa) * PI_GAMMA
        # Hakem geçmişi
        if isinstance(ref, str) and ref:
            cards = sum(float(getattr(r, c)) for c in ("hy", "ay", "hr", "ar")
                        if c in df.columns and not pd.isna(getattr(r, c))) if "hy" in df.columns else np.nan
            if "hy" in df.columns and pd.isna(r.hy):
                cards = np.nan
            pens = (float(r.hpen) + float(r.apen)) if "hpen" in df.columns and not pd.isna(r.hpen) else np.nan
            ref_hist[ref].append((cards, pens, float(hg > ag)))
        # Elo güncelle (gol farkı çarpanlı)
        res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        mult = np.log1p(abs(hg - ag)) + 1.0
        delta = ELO_K * mult * (res - e_h)
        elo[r.home] += delta
        elo[r.away] -= delta
        for t, p, gd in ((r.home, ph, hg - ag), (r.away, pa, ag - hg)):
            table[t]["p"] += p
            table[t]["gd"] += gd
            table[t]["n"] += 1
        h2h[key].append((r.date, r.home, hg - ag))

    return pd.DataFrame(out)


if __name__ == "__main__":
    f = build_features(load_matches())
    f.to_csv(ROOT / "data" / "features.csv", index=False)
    print(f.shape)
    print(f[f.played].iloc[-1].to_string())
