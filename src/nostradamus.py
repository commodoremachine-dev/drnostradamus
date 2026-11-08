"""Çevrimdışı, yalnızca sonuç verisi kullanan Dixon-Coles araştırma akışı.

python src/nostradamus.py test --baslangic 2024-07-01
python src/nostradamus.py tahmin Galatasaray Fenerbahçe --tarih 2026-11-08 --oran 1.95 3.60 4.00
"""
import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models import DixonColes, evaluate, markets, score_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "matches.csv"
DEFAULT_REPORT = ROOT / "reports" / "nostradamus"
PROBS = ["p_home", "p_draw", "p_away"]
ODDS = ["dec_h", "dec_d", "dec_a"]


def today():
    return pd.Timestamp(datetime.now(timezone(timedelta(hours=3))).date())


def read_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "home", "away", "hg", "ag", "played"}
    if required - set(df):
        raise ValueError(f"Eksik sütunlar: {sorted(required - set(df))}")
    if df[["date", "home", "away"]].isna().any().any():
        raise ValueError("Tarih ve takım adları boş olamaz.")
    flags = df.played.astype(str).str.lower()
    if not flags.isin(["true", "false", "1", "0"]).all():
        raise ValueError("played sütunu True/False veya 1/0 olmalı.")
    df["played"] = flags.isin(["true", "1"])
    for col in ["hg", "ag"] + [c for c in ODDS if c in df]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    goals = df.loc[df.played, ["hg", "ag"]].to_numpy(float)
    if not np.isfinite(goals).all() or (goals < 0).any() or (goals != np.floor(goals)).any():
        raise ValueError("Oynanmış maçların golleri negatif olmayan tam sayılar olmalı.")
    if df.duplicated(["date", "home", "away"]).any():
        raise ValueError("Aynı tarih/takımlar için birden fazla kayıt var.")
    if (df.home == df.away).any():
        raise ValueError("Bir takım kendisiyle oynayamaz.")
    return df.sort_values(["date", "home", "away"]).reset_index(drop=True)


def model_at(data, cutoff, half_life=365.0, l2=0.5):
    return DixonColes(xi=np.log(2) / half_life, l2=l2).fit(data, cutoff)


def forecast(model, home, away):
    lh, la = model.lambdas(home, away)
    mk = markets(score_matrix(lh, la, model.rho))
    mk["lh"], mk["la"] = float(lh), float(la)
    mk["top_score"] = mk.pop("top_scores")[0][0]
    return mk


def week_labels(dates):
    return pd.to_datetime(dates).dt.to_period("W-SUN").astype(str)


def bootstrap_probabilities(data, cutoff, home, away, draws, seed, half_life, l2):
    """Haftaları kümeler halinde yeniden örnekle, her örnekte modeli yeniden eğit.

    Yüzdelikler model/veri belirsizliği göstergesidir; kapsama garantili güven aralığı değildir.
    """
    train = data[data.played & (data.date < cutoff)
                 & (data.date >= cutoff - pd.Timedelta(days=365 * 4))].copy()
    groups = [g for _, g in train.groupby(week_labels(train.date))]
    if len(groups) < 20:
        raise ValueError("Belirsizlik hesabı için en az 20 maç haftası gerekir.")
    rng, probs = np.random.default_rng(seed), []
    for _ in range(draws):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        model = model_at(sample, cutoff, half_life, l2)
        mk = forecast(model, home, away)
        probs.append([mk[c] for c in PROBS])
    return np.asarray(probs)


def research_decisions(p, lower, odds, bankroll=0.0, min_ev=0.05, eligible=True):
    p, lower, odds = np.asarray(p), np.asarray(lower), np.asarray(odds)
    if (p.shape != (3,) or lower.shape != (3,) or odds.shape != (3,)
            or not np.isfinite(odds).all() or (odds <= 1).any()
            or not np.isfinite(p).all() or not np.isfinite(lower).all()
            or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1)
            or (lower < 0).any() or (lower > 1).any()
            or not np.isfinite(bankroll) or bankroll < 0
            or not np.isfinite(min_ev) or min_ev < 0):
        raise ValueError("Olasılıklar geçerli, ondalık oranlar 1'den büyük, kasa negatif olmayan sayı olmalı.")
    conservative = np.minimum(p, lower)
    lower_ev = conservative * odds - 1
    # Bir maçta en fazla bir seçim; ilişkili bahisler için ayrı Kelly uygulanmaz.
    best = int(np.argmax(lower_ev))
    rows = []
    for i, label in enumerate(["1", "X", "2"]):
        selected = eligible and i == best and lower_ev[i] > min_ev
        fraction = min(0.005, 0.25 * max(0.0, lower_ev[i] / (odds[i] - 1))) if selected else 0.0
        rows.append(dict(secim=label, olasilik=float(p[i]), alt_tahmin=float(conservative[i]),
                         adil_oran=float(1 / p[i]) if p[i] > 0 else None,
                         oran=float(odds[i]), ev=float(p[i] * odds[i] - 1),
                         alt_ev=float(lower_ev[i]), karar="ARAŞTIRMA ADAYI" if selected else "BAHİS YOK",
                         simulasyon_kasa_payi=float(fraction), simulasyon_tutari=float(bankroll * fraction)))
    return rows


def probability_metrics(p, y):
    p = np.asarray(p, float)
    return dict(n=len(y), logloss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
                brier=float(((p - np.eye(3)[y]) ** 2).sum(1).mean()),
                accuracy=float((p.argmax(1) == y).mean()))


def weekly_roi_interval(bets, draws=2000, seed=42):
    if bets.empty:
        return None
    g = bets.groupby(week_labels(bets.date)).profit.agg(["sum", "count"])
    if len(g) < 2:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(g), (draws, len(g)))
    roi = g["sum"].to_numpy()[indices].sum(1) / g["count"].to_numpy()[indices].sum(1)
    return [float(x) for x in np.quantile(roi, [0.025, 0.975])]


def walk_forward(data, start, end, half_life, l2, refit_days=30):
    target = data[data.played & data.date.between(start, end)]
    rows, model, last_fit = [], None, None
    for row in target.itertuples(index=False):
        if model is None or (row.date - last_fit).days >= refit_days:
            model = model_at(data, row.date, half_life, l2)
            if model.train_count < 200:
                raise ValueError("İlk test maçından önce en az 200 eğitim maçı gerekir.")
            last_fit = row.date
        prediction = forecast(model, row.home, row.away)
        rows.append(dict(date=row.date, home=row.home, away=row.away, hg=row.hg, ag=row.ag,
                         train_end=model.train_end, train_count=model.train_count,
                         **prediction, **{c: getattr(row, c, np.nan) for c in ODDS}))
    if not rows:
        raise ValueError("Seçilen dönemde oynanmış test maçı yok.")
    return pd.DataFrame(rows)


def backtest_report(pred, min_ev):
    p = pred[PROBS].to_numpy()
    y = np.where(pred.hg > pred.ag, 0, np.where(pred.hg == pred.ag, 1, 2))
    odds = pred[ODDS].to_numpy(float)
    valid = np.isfinite(odds).all(1) & (odds > 1).all(1)
    report = {"model": evaluate(pred), "quote_timing_verified": False,
              "interpretation": "Keşifsel tekrar testidir; oran zamanları doğrulanmadı, yeni ve dokunulmamış test değildir."}
    bets = pred.iloc[0:0][["date", "home", "away"]].copy()
    if valid.any():
        q = 1 / odds[valid]
        q /= q.sum(1, keepdims=True)
        report["matched_model"] = probability_metrics(p[valid], y[valid])
        report["matched_market"] = probability_metrics(q, y[valid])
        edges = p[valid] * odds[valid] - 1
        selection = edges.argmax(1)
        keep = edges[np.arange(len(edges)), selection] > min_ev
        bets = pred.loc[valid, ["date", "home", "away"]].iloc[np.flatnonzero(keep)].copy()
        picked = selection[keep]
        chosen_odds = odds[valid][keep, picked]
        bets["selection"] = np.array(["1", "X", "2"])[picked]
        bets["odds"] = chosen_odds
        bets["predicted_ev"] = edges[keep, picked]
        bets["profit"] = np.where(picked == y[valid][keep], chosen_odds - 1, -1.0)
    if len(bets):
        path = np.r_[0.0, bets.groupby("date").profit.sum().cumsum().to_numpy()]
        report["exploratory_strategy"] = dict(bets=len(bets), flat_stake_units=1,
            profit_units=float(bets.profit.sum()), roi=float(bets.profit.mean()),
            weekly_bootstrap_roi_95=weekly_roi_interval(bets),
            max_drawdown_units=float((np.maximum.accumulate(path) - path).max()),
            rule=f"Maç başına en yüksek EV; EV > {min_ev}; belirsizlik filtresi uygulanmadı.")
    else:
        report["exploratory_strategy"] = {"bets": 0, "roi": None}
    calibration = []
    for k, label in enumerate(["1", "X", "2"]):
        bins = np.minimum((p[:, k] * 10).astype(int), 9)
        for b in np.unique(bins):
            mask = bins == b
            calibration.append(dict(selection=label, bin=int(b), n=int(mask.sum()),
                                    predicted=float(p[mask, k].mean()), actual=float((y[mask] == k).mean())))
    return report, bets, pd.DataFrame(calibration)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(args):
    data = read_data(args.veri)
    args.cikti.mkdir(parents=True, exist_ok=True)
    common = dict(data_sha256=hashlib.sha256(args.veri.read_bytes()).hexdigest(),
                  half_life_days=args.yari_omur, l2=args.l2, min_ev=args.esik,
                  generated_at=datetime.now(timezone.utc).isoformat())
    if args.command == "test":
        start, end = pd.Timestamp(args.baslangic), min(pd.Timestamp(args.bitis), today() - pd.Timedelta(days=1))
        pred = walk_forward(data, start, end, args.yari_omur, args.l2, args.yenile)
        report, bets, calibration = backtest_report(pred, args.esik)
        report["configuration"] = dict(**common, start=str(start.date()), end=str(end.date()), refit_days=args.yenile)
        pred.to_csv(args.cikti / "predictions.csv", index=False)
        bets.to_csv(args.cikti / "bets.csv", index=False)
        calibration.to_csv(args.cikti / "calibration.csv", index=False)
        write_json(args.cikti / "summary.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        fixture_date = pd.Timestamp(args.tarih).normalize()
        cutoff = min(fixture_date, today())
        model = model_at(data, cutoff, args.yari_omur, args.l2)
        known = sorted(set(data.home) | set(data.away))
        from teams import normalize
        names = []
        for name in [args.ev, args.deplasman]:
            match = next((t for t in known if t.casefold() == normalize(name).casefold()), None)
            if match is None:
                raise ValueError(f"Takım bulunamadı: {name}")
            names.append(match)
        home, away = names
        if home == away:
            raise ValueError("İki farklı takım seçin.")
        mk = forecast(model, home, away)
        print(f"Belirsizlik için {args.ornek} model yeniden eğitiliyor...", flush=True)
        samples = bootstrap_probabilities(data, cutoff, home, away, args.ornek, args.tohum, args.yari_omur, args.l2)
        lower = np.quantile(samples, 0.05, axis=0)
        train = data[data.played & (data.date < cutoff) & (data.date >= cutoff - pd.Timedelta(days=365 * 4))]
        counts = {t: int(((train.home == t) | (train.away == t)).sum()) for t in names}
        fresh = all((cutoff - train.loc[(train.home == t) | (train.away == t), "date"].max()).days <= 60
                    for t in names if counts[t])
        eligible = model.train_count >= 200 and min(counts.values()) >= 5 and fresh
        report = dict(configuration=common, home=home, away=away, fixture_date=str(fixture_date.date()),
                      train_end=str(model.train_end), train_count=model.train_count, team_matches=counts,
                      eligible=eligible, bootstrap_draws=args.ornek, seed=args.tohum,
                      uncertainty="Haftalık yeniden örnekleme %5 yüzdeliği; kapsama garantili güven aralığı değildir.",
                      status="Araştırma/sanal kasa; doğrulanmış kârlılık yok.", forecast=mk,
                      decisions=research_decisions([mk[c] for c in PROBS], lower, args.oran,
                                                   args.kasa, args.esik, eligible))
        write_json(args.cikti / "forecast.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description="NOSTRADAMUS: çevrimdışı matematiksel araştırma modeli")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["test", "tahmin"]:
        p = sub.add_parser(command)
        p.add_argument("--veri", type=Path, default=DEFAULT_DATA)
        p.add_argument("--cikti", type=Path, default=DEFAULT_REPORT)
        p.add_argument("--yari-omur", type=float, default=365.0)
        p.add_argument("--l2", type=float, default=0.5)
        p.add_argument("--esik", type=float, default=0.05)
        if command == "test":
            p.add_argument("--baslangic", default="2024-07-01")
            p.add_argument("--bitis", default=str(today().date()))
            p.add_argument("--yenile", type=int, default=30)
        else:
            p.add_argument("ev")
            p.add_argument("deplasman")
            p.add_argument("--tarih", required=True)
            p.add_argument("--oran", type=float, nargs=3, required=True, metavar=("EV", "X", "DEP"))
            p.add_argument("--ornek", type=int, default=100)
            p.add_argument("--tohum", type=int, default=42)
            p.add_argument("--kasa", type=float, default=0.0, help="İsteğe bağlı sanal kasa; varsayılan 0")
    args = parser.parse_args()
    if (not np.isfinite(args.yari_omur) or args.yari_omur <= 0
            or not np.isfinite(args.l2) or args.l2 < 0
            or not np.isfinite(args.esik) or args.esik < 0):
        parser.error("Yarı ömür pozitif; l2 ve eşik negatif olmayan sonlu sayılar olmalı.")
    if args.command == "test" and args.yenile < 1:
        parser.error("Yenileme aralığı en az 1 gün olmalı.")
    if args.command == "tahmin" and (args.ornek < 30 or not np.isfinite(args.kasa) or args.kasa < 0
            or not np.isfinite(args.oran).all() or min(args.oran) <= 1 or args.tohum < 0):
        parser.error("En az 30 örnek; sonlu, negatif olmayan kasa; 1'den büyük sonlu oranlar ve negatif olmayan tohum gerekir.")
    try:
        run(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"Hata: {exc}\n")


if __name__ == "__main__":
    main()
