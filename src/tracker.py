"""Tahminleri canlı sonuçlarla karşılaştırır.

  python src/tracker.py kaydet [--gun 7]   # başlamamış maçların tahminlerini kaydet
  python src/tracker.py durum              # canlı skorlar + biten maçların değerlendirmesi
  python src/tracker.py                    # ikisi birden

Kurallar (dürüst ölçüm için):
  - Tahmin yalnızca maç BAŞLAMADAN önce kaydedilir (ESPN durumu "pre").
  - Maç başlamadan tekrar çalıştırılırsa kayıt en güncel tahminle yenilenir
    (ilk kayıt zamanı da saklanır); maç başladıktan sonra kayıt kilitlenir.
Kayıt dosyası: data/predictions_log.csv
"""
import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from espn_source import BASE, _get  # noqa: E402
from teams import normalize  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "predictions_log.csv"


def live_events(year=None):
    """ESPN'den bu yılın tüm maçlarını canlı durumlarıyla çeker (tek istek)."""
    year = year or datetime.now().year
    out = []
    for e in _get(f"{BASE}/scoreboard?dates={year}&limit=1000").get("events", []):
        c = e["competitions"][0]
        st = c["status"]["type"]
        sides = {x["homeAway"]: x for x in c["competitors"]}
        kick = pd.Timestamp(e["date"])
        out.append(dict(
            espn_id=str(e["id"]), kickoff_utc=kick.tz_convert("UTC").tz_localize(None),
            date=(kick.tz_convert("UTC") + pd.Timedelta(hours=3)).normalize().tz_localize(None),
            season=f"{e['season']['year']}-{(e['season']['year'] + 1) % 100:02d}",
            home=normalize(sides["home"]["team"]["displayName"]),
            away=normalize(sides["away"]["team"]["displayName"]),
            state=st.get("state"), completed=bool(st.get("completed")), detail=st.get("shortDetail"),
            status_name=st.get("name"),
            hs=int(sides["home"]["score"]) if st.get("state") != "pre" else None,
            as_=int(sides["away"]["score"]) if st.get("state") != "pre" else None))
    return pd.DataFrame(out)


def load_log() -> pd.DataFrame:
    if LOG.exists():
        # "1"/"2" seçimleri sayıya dönüşmesin (yoksa isabet karşılaştırması bozulur)
        return pd.read_csv(LOG, dtype={"espn_id": str, "pick": str, "value_pick": str, "top_score": str},
                           parse_dates=["kickoff_utc", "date"])
    return pd.DataFrame()


def cmd_kaydet(days: int):
    from features import load_matches
    from predict import match_markets, predict, upset_info

    ev = live_events()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    todo = ev[(ev.state == "pre") & (ev.kickoff_utc > now) &
              (ev.kickoff_utc <= now + pd.Timedelta(days=days))]
    if todo.empty:
        print("Önümüzdeki günlerde başlamamış maç yok.")
        return
    matches = load_matches()
    fx = todo[["season", "date", "home", "away"]].copy()
    res = predict(fx.assign(round=pd.NA, hg=pd.NA, ag=pd.NA, hthg=pd.NA, htag=pd.NA, played=False), matches)
    res = res.merge(todo[["date", "home", "away", "espn_id", "kickoff_utc"]], on=["date", "home", "away"])
    from nesine import find_odds
    from value import VALUE_MIN_EV, value_table
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    odds_cols = [c for c in ("dec_h", "dec_d", "dec_a", "dec_over25", "dec_under25") if c in matches.columns]
    odds_map = matches.set_index(["season", "home", "away"])[odds_cols]
    rows = []
    for r in res.itertuples(index=False):
        m, mk = match_markets(r)
        u = upset_info(r) or {}
        key = (r.season, r.home, r.away)
        od = odds_map.loc[key].to_dict() if key in odds_map.index else {}
        if isinstance(od.get("dec_h"), (int, float)) and od.get("dec_h") == od.get("dec_h"):
            o1x2 = (od["dec_h"], od["dec_d"], od["dec_a"])
        else:
            o1x2 = None
        oou = (od["dec_over25"], od["dec_under25"]) if od.get("dec_over25") == od.get("dec_over25") \
            and od.get("dec_over25") is not None else None
        # Bültendeki maçlarda oynanabilir iddaa oranı (Nesine) kullanılır; yoksa ESPN/Bet365
        odds_src = "ESPN/Bet365" if o1x2 else None
        nz = find_odds(r.date, r.home, r.away)
        if nz and nz["o1x2"]:
            o1x2, oou, odds_src = nz["o1x2"], nz["oou"] or oou, "Nesine"
        vt = value_table(r, o1x2, oou)
        best = vt.dropna(subset=["ev_adil"]).sort_values("ev_adil", ascending=False)
        best = best[best.ev_adil > VALUE_MIN_EV].head(1)
        pick = ["1", "X", "2"][int(np.argmax([mk["p_home"], mk["p_draw"], mk["p_away"]]))]
        rows.append(dict(
            espn_id=r.espn_id, kickoff_utc=r.kickoff_utc, date=r.date, home=r.home, away=r.away,
            logged_at=stamp, first_logged_at=stamp,
            lh=round(r.lh, 3), la=round(r.la, 3),
            p_home=round(mk["p_home"], 4), p_draw=round(mk["p_draw"], 4), p_away=round(mk["p_away"], 4),
            p_over25=round(mk["p_over25"], 4), p_btts=round(mk["p_btts"], 4),
            pick=pick, top_score=mk["top_scores"][0][0],
            upset_prob=round(u.get("surpriz_olasiligi", np.nan), 4), upset_level=u.get("seviye"),
            fav=(r.home if u.get("fav_is_home") else r.away) if u else None,
            mkt_home=getattr(r, "odds_ph", np.nan), mkt_draw=getattr(r, "odds_pd", np.nan),
            mkt_away=getattr(r, "odds_pa", np.nan),
            dec_h=o1x2[0] if o1x2 else np.nan, dec_d=o1x2[1] if o1x2 else np.nan,
            dec_a=o1x2[2] if o1x2 else np.nan,
            value_pick=best.secim.iloc[0] if len(best) else None,
            value_odds=best.oran.iloc[0] if len(best) else np.nan,
            value_ev=round(best.ev_adil.iloc[0], 4) if len(best) else np.nan,
            odds_src=odds_src))
    new = pd.DataFrame(rows)
    log = load_log()
    if not log.empty:
        # Kilitli kayıtlar: maçı başlamış/bitmiş olanlar ya da bu çalıştırmada yeniden tahmin edilmeyenler
        firsts = log.set_index("espn_id")["first_logged_at"]
        new["first_logged_at"] = new.espn_id.map(firsts).fillna(new.first_logged_at)
        log = log[~log.espn_id.isin(new.espn_id)]
        new = pd.concat([log, new], ignore_index=True)
    new.sort_values("kickoff_utc").to_csv(LOG, index=False)
    print(f"{len(rows)} maçın tahmini kaydedildi ({stamp} UTC) -> {LOG.name}")
    for r in sorted(rows, key=lambda x: x["kickoff_utc"]):
        print(f"  {pd.Timestamp(r['kickoff_utc']) + pd.Timedelta(hours=3):%d.%m %H:%M}  {r['home']} - {r['away']}: "
              f"tahmin {r['pick']} (%{100 * max(r['p_home'], r['p_draw'], r['p_away']):.0f}), skor {r['top_score']}, "
              f"sürpriz alarmı {r['upset_level']}"
              + (f", VALUE: {r['value_pick']} @ {r['value_odds']:.2f} (EV %{100 * r['value_ev']:+.1f})"
                 if r.get("value_pick") else ""))


def _outcome(h, a):
    return "1" if h > a else "X" if h == a else "2"


def status_frames():
    """Arayüz için: kayıtlı tahminler + canlı durum, biten maçların isabet sütunları ve özet.

    Döner: (log, done, ozet)  — log boşsa (boş, boş, {}).
    """
    log = load_log()
    if log.empty:
        return log, log, {}
    try:
        ev = live_events().set_index("espn_id")
        log = log.join(ev[["state", "completed", "detail", "hs", "as_", "status_name"]], on="espn_id")
    except Exception:  # internet yoksa yalnızca kayıtları göster
        for c in ("state", "completed", "detail", "hs", "as_", "status_name"):
            log[c] = None
    log["kickoff_tr"] = pd.to_datetime(log.kickoff_utc) + pd.Timedelta(hours=3)
    done = log[log.completed == True].copy()  # noqa: E712
    ozet = {}
    if done.empty:
        return log, done, ozet
    done["res"] = [_outcome(h, a) for h, a in zip(done.hs, done.as_)]
    done["hit_1x2"] = done.res == done.pick
    done["hit_score"] = done.top_score == done.hs.astype(int).astype(str) + "-" + done.as_.astype(int).astype(str)
    done["hit_o25"] = (done.p_over25 > 0.5) == ((done.hs + done.as_) > 2)
    done["fav_lost"] = [(r.fav == r.home and r.res == "2") or (r.fav == r.away and r.res == "1")
                        for r in done.itertuples()]
    ozet = {"mac": len(done), "isabet_1x2": done.hit_1x2.mean(), "isabet_skor": done.hit_score.mean(),
            "isabet_o25": done.hit_o25.mean(), "favori_kaybetti": int(done.fav_lost.sum())}
    if "value_pick" in done.columns:
        tot = done.hs + done.as_
        won = [(isinstance(vp, str)) and ((vp in ("1", "X", "2") and vp == rs) or (vp == "2.5 Üst" and t > 2)
                                         or (vp == "2.5 Alt" and t <= 2))
               for vp, rs, t in zip(done.value_pick, done.res, tot)]
        done["value_kar"] = np.where(done.value_pick.notna(), np.where(won, done.value_odds - 1, -1.0), np.nan)
        vb = done.value_kar.dropna()
        if len(vb):
            ozet.update(value_bahis=len(vb), value_kar=vb.sum(), value_roi=vb.mean())
    return log, done, ozet


def cmd_durum():
    log = load_log()
    if log.empty:
        print("Henüz kayıtlı tahmin yok. Önce: python src/tracker.py kaydet")
        return
    ev = live_events().set_index("espn_id")
    log = log.join(ev[["state", "completed", "detail", "hs", "as_", "status_name"]], on="espn_id")
    log["kickoff_tr"] = pd.to_datetime(log.kickoff_utc) + pd.Timedelta(hours=3)

    live = log[log.state == "in"]
    if len(live):
        print("CANLI MAÇLAR")
        for r in live.itertuples():
            now_pick = _outcome(r.hs, r.as_)
            print(f"  {r.home} {int(r.hs)}-{int(r.as_)} {r.away}  [{r.detail}]  | tahmin {r.pick} "
                  f"(1 %{100 * r.p_home:.0f} / X %{100 * r.p_draw:.0f} / 2 %{100 * r.p_away:.0f}), "
                  f"skor {r.top_score} | şu an {'TUTUYOR' if now_pick == r.pick else 'tutmuyor'}")
    pre = log[log.state == "pre"]
    if len(pre):
        print("\nBAŞLAMAMIŞ MAÇLAR")
        for r in pre.itertuples():
            print(f"  {r.kickoff_tr:%d.%m %H:%M}  {r.home} - {r.away}: tahmin {r.pick}, skor {r.top_score}, "
                  f"sürpriz {r.upset_level}")

    done = log[log.completed == True].copy()  # noqa: E712
    if done.empty:
        print("\nHenüz biten kayıtlı maç yok; değerlendirme maçlar bittikçe birikecek.")
        return
    done["res"] = [_outcome(h, a) for h, a in zip(done.hs, done.as_)]
    done["hit_1x2"] = done.res == done.pick
    done["hit_score"] = done.top_score == done.hs.astype(int).astype(str) + "-" + done.as_.astype(int).astype(str)
    done["over25"] = (done.hs + done.as_) > 2
    done["hit_o25"] = (done.p_over25 > 0.5) == done.over25
    idx = done.res.map({"1": 0, "X": 1, "2": 2}).values
    P = done[["p_home", "p_draw", "p_away"]].values
    done["logloss"] = -np.log(np.clip(P[np.arange(len(done)), idx], 1e-9, 1))
    fav_lost = [(r.fav == r.home and r.res == "2") or (r.fav == r.away and r.res == "1") for r in done.itertuples()]
    done["fav_lost"] = fav_lost

    print(f"\nBİTEN MAÇLAR ({len(done)})")
    for r in done.sort_values("kickoff_utc").itertuples():
        vtxt = f" | value {r.value_pick}@{r.value_odds:.2f}" if getattr(r, "value_pick", None) == getattr(r, "value_pick", None) \
            and isinstance(getattr(r, "value_pick", None), str) else ""
        print(f"  {r.kickoff_tr:%d.%m}  {r.home} {int(r.hs)}-{int(r.as_)} {r.away}{vtxt} | tahmin {r.pick} "
              f"{'✓' if r.hit_1x2 else '✗'} | skor {r.top_score} {'✓' if r.hit_score else '✗'} | "
              f"2.5 {'Ü' if r.p_over25 > 0.5 else 'A'} {'✓' if r.hit_o25 else '✗'} | sürpriz {r.upset_level}"
              f"{' → FAVORİ KAYBETTİ' if r.fav_lost else ''}")
    n = len(done)
    print(f"\nÖZET ({n} maç)")
    print(f"  Maç sonucu (1X2) isabeti : %{100 * done.hit_1x2.mean():.1f}   (geçmiş testte beklenen ~%54)")
    print(f"  Tam skor isabeti         : %{100 * done.hit_score.mean():.1f}   (beklenen ~%12)")
    print(f"  2.5 alt/üst isabeti      : %{100 * done.hit_o25.mean():.1f}   (beklenen ~%55)")
    print(f"  Log loss                 : {done.logloss.mean():.3f}  (beklenen ~0.98, düşük = iyi)")
    mk = done.dropna(subset=["mkt_home"])
    if len(mk):
        mp = mk[["mkt_home", "mkt_draw", "mkt_away"]].values.argmax(1)
        mres = mk.res.map({"1": 0, "X": 1, "2": 2}).values
        print(f"  Bahis piyasası isabeti   : %{100 * (mp == mres).mean():.1f}  ({len(mk)} oranlı maç; model aynı maçlarda "
              f"%{100 * mk.hit_1x2.mean():.1f})")
    if "value_pick" in done.columns:
        vb = done[done.value_pick.notna()].copy()
        if len(vb):
            tot = vb.hs + vb.as_
            won = [(vp == "1" and rs == "1") or (vp == "X" and rs == "X") or (vp == "2" and rs == "2")
                   or (vp == "2.5 Üst" and t > 2) or (vp == "2.5 Alt" and t <= 2)
                   for vp, rs, t in zip(vb.value_pick, vb.res, tot)]
            vb["kar"] = np.where(won, vb.value_odds - 1, -1.0)
            print(f"  Value bet (1 birim)      : {len(vb)} bahis, {int(sum(won))} kazandı, "
                  f"kâr {vb.kar.sum():+.2f} birim, ROI %{100 * vb.kar.mean():+.1f}")
    print(f"  Favori kaybetti          : {int(done.fav_lost.sum())} maç")
    for lv in ["düşük", "orta", "yüksek", "çok yüksek"]:
        g = done[done.upset_level == lv]
        if len(g):
            print(f"    alarm {lv:10s}: {len(g):3d} maç, favori kaybetti %{100 * g.fav_lost.mean():.0f}")
    if n < 50:
        print(f"  Not: {n} maç az bir örnek; oranlar ±%{100 * 1.96 * np.sqrt(0.25 / n):.0f} oynayabilir.")
    done.drop(columns=["state", "completed", "detail", "status_name"]).to_csv(
        ROOT / "reports" / "live_evaluation.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("komut", nargs="?", choices=["kaydet", "durum"], default=None)
    ap.add_argument("--gun", type=int, default=7)
    a = ap.parse_args()
    if a.komut in (None, "kaydet"):
        cmd_kaydet(a.gun)
    if a.komut in (None, "durum"):
        print()
        cmd_durum()


if __name__ == "__main__":
    main()
