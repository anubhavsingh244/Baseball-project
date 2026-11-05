import unittest

from baseball import data, stats


class StatsTests(unittest.TestCase):
    def test_avg(self):
        self.assertAlmostEqual(stats.batting_average(3, 10), 0.3)
        self.assertEqual(stats.batting_average(0, 0), 0.0)

    def test_obp_slg_ops(self):
        # 100 AB, 30 H (6 2B, 1 3B, 5 HR), 10 BB, 2 HBP, 3 SF
        self.assertAlmostEqual(stats.on_base_percentage(30, 10, 2, 100, 3), 42 / 115)
        self.assertAlmostEqual(stats.slugging(30, 6, 1, 5, 100), 0.53)
        self.assertAlmostEqual(stats.ops(30, 6, 1, 5, 10, 2, 100, 3), 42 / 115 + 0.53)

    def test_innings(self):
        self.assertAlmostEqual(stats.innings_to_float(195.1), 195 + 1 / 3)
        self.assertAlmostEqual(stats.innings_to_float(180.2), 180 + 2 / 3)

    def test_era_whip(self):
        self.assertAlmostEqual(stats.era(10, 90), 1.0)
        self.assertAlmostEqual(stats.whip(80, 20, 100), 1.0)

    def test_sample_data(self):
        batting = data.load_batting()
        self.assertEqual(len(batting), 10)
        career = {r["player"]: r for r in data.career_batting(batting)}
        self.assertEqual(career["Alex Rivera"]["HR"], 60)
        self.assertTrue(all(0 <= r["AVG"] <= 1 for r in batting))
        self.assertEqual(len(data.load_pitching()), 6)


if __name__ == "__main__":
    unittest.main()
