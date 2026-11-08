"""Model güvenliği ve zaman sınırı için regresyon testleri (ağ erişimi yok)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import DixonColes, markets, score_matrix, valid_rho
from nostradamus import backtest_report, research_decisions, walk_forward


def fixture_data():
    rng = np.random.default_rng(8)
    rows = []
    for i in range(260):
        home, away = ("A", "B") if i % 2 else ("B", "A")
        rows.append(dict(date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                         home=home, away=away, hg=int(rng.poisson(1.5)), ag=int(rng.poisson(1.1)),
                         played=True, dec_h=2.0, dec_d=3.3, dec_a=3.8))
    return pd.DataFrame(rows)


class ScoreSafetyTests(unittest.TestCase):
    def test_extreme_rho_cannot_create_negative_probability(self):
        for lh, la in [(8.0, 6.0), (0.1, 12.0), (1.4, 1.2)]:
            for rho in [-0.9, 0.9]:
                m = score_matrix(lh, la, rho)
                self.assertTrue(np.isfinite(m).all())
                self.assertGreaterEqual(m.min(), 0)
                self.assertAlmostEqual(m.sum(), 1)
                mk = markets(m)
                self.assertAlmostEqual(sum(mk[k] for k in ["p_home", "p_draw", "p_away"]), 1)

    def test_dynamic_grid_preserves_large_poisson_mean(self):
        m = score_matrix(14.0, 11.0)
        self.assertGreater(m.shape[0], 10)
        self.assertAlmostEqual(markets(m)["xg_home"], 14.0, places=6)
        self.assertAlmostEqual(markets(m)["xg_away"], 11.0, places=6)

    def test_bad_rates_are_rejected(self):
        for rate in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                score_matrix(rate, 1.0)

    def test_rho_bounds_apply_to_unobserved_low_scores_too(self):
        lh, la = np.array([8., 1.]), np.array([6., 2.])
        for candidate in [-1, 1]:
            rho = valid_rho(lh, la, candidate)
            self.assertTrue((1 - lh * la * rho > 0).all())
            self.assertTrue((1 + lh * rho > 0).all())
            self.assertTrue((1 + la * rho > 0).all())
            self.assertGreater(1 - rho, 0)


class ChronologyTests(unittest.TestCase):
    def test_future_and_same_day_results_cannot_affect_fit(self):
        data = fixture_data()
        cutoff = data.date.iloc[210]
        before = DixonColes().fit(data, cutoff)
        changed = data.copy()
        changed.loc[changed.date >= cutoff, ["hg", "ag"]] = [40, 30]
        after = DixonColes().fit(changed, cutoff)
        np.testing.assert_allclose(before.lambdas("A", "B"), after.lambdas("A", "B"))
        self.assertLess(before.train_end, cutoff)
        self.assertEqual(before.train_count, 210)
        self.assertAlmostEqual(sum(before.att.values()), 0, places=8)
        self.assertAlmostEqual(sum(before.dfn.values()), 0, places=8)

    def test_walkforward_never_trains_on_target_dates(self):
        data = fixture_data()
        p = walk_forward(data, data.date.iloc[210], data.date.iloc[-1], 365, .5, 10)
        self.assertTrue((p.train_end < p.date).all())
        report, bets, calibration = backtest_report(p, 0.05)
        self.assertFalse(report["quote_timing_verified"])
        self.assertEqual(report["matched_market"]["n"], len(p))
        self.assertFalse(bets.duplicated(["date", "home", "away"]).any())
        self.assertTrue((calibration.n > 0).all())


class DecisionTests(unittest.TestCase):
    def test_uncertainty_can_veto_positive_point_ev(self):
        rows = research_decisions([.55, .25, .20], [.45, .18, .14], [1.95, 3.5, 4.0], 1000)
        self.assertGreater(rows[0]["ev"], 0)
        self.assertTrue(all(r["simulasyon_tutari"] == 0 for r in rows))

    def test_maximum_one_selection_and_half_percent_cap(self):
        rows = research_decisions([.55, .25, .20], [.54, .24, .19], [3.0, 8.0, 10.0], 1000)
        self.assertEqual(sum(r["simulasyon_tutari"] > 0 for r in rows), 1)
        self.assertLessEqual(sum(r["simulasyon_tutari"] for r in rows), 5.0)
        rows = research_decisions([.55, .25, .20], [.54, .24, .19], [3.0, 8.0, 10.0], 1000, eligible=False)
        self.assertTrue(all(r["karar"] == "BAHİS YOK" for r in rows))

    def test_invalid_odds_are_rejected(self):
        for odds in [[1, 3, 4], [np.nan, 3, 4]]:
            with self.assertRaises(ValueError):
                research_decisions([.55, .25, .20], [.54, .24, .19], odds)


class NesineMatchTests(unittest.TestCase):
    """Takım adı eşleştirme (ağ erişimi yok: bülten önbelleğe sahte veriyle konur)."""

    def setUp(self):
        import nesine
        self.n = nesine
        ms = lambda a, b, c: dict(MT=1, MST=1, SOV=0.0, MBS=1, OCA=[dict(N=1, O=a), dict(N=2, O=b), dict(N=3, O=c)])
        au = dict(MT=2, MST=101, SOV=2.5, MBS=1, OCA=[dict(N=1, O=1.95), dict(N=2, O=1.48)])
        ev = [dict(C=1, D="08.11.2026", T="13:30", HN="Rizespor", AN="Erzurumspor FK", MA=[ms(1.64, 3.3, 3.65), au]),
              dict(C=2, D="08.11.2026", T="11:00", HN="Fenerbahçe U19", AN="Çorum FK U19", MA=[ms(1.2, 5, 9)]),
              dict(C=3, D="08.11.2026", T="18:00", HN="Modern Sport Club", AN="Alqana", MA=[ms(2, 3, 3)])]
        nesine._cache.update(t=1e18, events=ev)  # önbellek hiç eskimesin

    def test_sponsor_and_suffix_names_match(self):
        r = self.n.find_odds("2026-11-08", "Çaykur Rizespor", "Erzurumspor")
        self.assertEqual(r["o1x2"], (1.64, 3.3, 3.65))
        self.assertEqual(r["oou"], (1.48, 1.95))  # (Üst, Alt)

    def test_youth_team_is_not_matched(self):
        self.assertIsNone(self.n.find_odds("2026-11-08", "Fenerbahçe", "Çorum"))

    def test_unrelated_team_is_not_matched(self):
        self.assertIsNone(self.n.find_odds("2026-11-08", "Alanyaspor", "Trabzonspor"))

    def test_other_day_is_not_matched(self):
        self.assertIsNone(self.n.find_odds("2026-11-21", "Çaykur Rizespor", "Erzurumspor"))


if __name__ == "__main__":
    unittest.main()
