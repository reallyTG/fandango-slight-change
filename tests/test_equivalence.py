#!/usr/bin/env pytest
"""Unit tests for the pure equivalence-test statistics (statistics/equivalence.py).

All bootstrap calls pass a fixed seed, so these are deterministic without relying on
PYTHONHASHSEED (the module uses a private random.Random, not global state)."""

import math
import statistics
import unittest

from fandango.statistics.equivalence import (
    discretization_floor,
    familywise_alpha,
    is_delta_below_floor,
    ks_statistic,
    ks_upper_ci,
    wasserstein_distance,
    wasserstein_upper_ci,
)


class TestWassersteinDistance(unittest.TestCase):
    def test_zero_on_target_quantiles(self):
        nd = statistics.NormalDist(30, 5)
        n = 200
        on_target = [nd.inv_cdf((i + 0.5) / n) for i in range(n)]
        self.assertAlmostEqual(wasserstein_distance(on_target, nd.inv_cdf), 0.0, places=6)

    def test_shift_moves_distance_by_shift(self):
        nd = statistics.NormalDist(0, 1)
        n = 200
        on_target = [nd.inv_cdf((i + 0.5) / n) for i in range(n)]
        shifted = [v + 3 for v in on_target]
        self.assertAlmostEqual(wasserstein_distance(shifted, nd.inv_cdf), 3.0, places=2)

    def test_empty_is_zero(self):
        self.assertEqual(wasserstein_distance([], lambda p: 0.0), 0.0)


class TestWassersteinUpperCI(unittest.TestCase):
    def test_upper_bound_exceeds_point_estimate(self):
        # For a real iid draw the bootstrap upper bound must sit above the point estimate.
        nd = statistics.NormalDist(30, 5)
        rng = __import__("random").Random(1)
        sample = [nd.inv_cdf(rng.random()) for _ in range(200)]
        point = wasserstein_distance(sample, nd.inv_cdf)
        upper = wasserstein_upper_ci(sample, nd.inv_cdf, confidence=0.95, seed=0)
        self.assertGreaterEqual(upper, point)

    def test_is_reproducible_with_seed(self):
        nd = statistics.NormalDist(0, 1)
        sample = [nd.inv_cdf((i + 0.5) / 50) for i in range(50)]
        a = wasserstein_upper_ci(sample, nd.inv_cdf, seed=7)
        b = wasserstein_upper_ci(sample, nd.inv_cdf, seed=7)
        self.assertEqual(a, b)

    def test_higher_confidence_is_wider(self):
        nd = statistics.NormalDist(0, 1)
        rng = __import__("random").Random(2)
        sample = [nd.inv_cdf(rng.random()) for _ in range(100)]
        lo = wasserstein_upper_ci(sample, nd.inv_cdf, confidence=0.80, seed=0)
        hi = wasserstein_upper_ci(sample, nd.inv_cdf, confidence=0.99, seed=0)
        self.assertGreaterEqual(hi, lo)

    def test_empty_is_zero(self):
        self.assertEqual(wasserstein_upper_ci([], lambda p: 0.0), 0.0)


class TestKS(unittest.TestCase):
    def test_ks_zero_for_perfect_uniform(self):
        n = 1000
        sample = [(i + 0.5) / n for i in range(n)]
        # empirical CDF of an evenly spaced grid vs Uniform(0,1) CDF is <= 1/n
        self.assertLessEqual(ks_statistic(sample, lambda x: x), 1.0 / n + 1e-9)

    def test_ks_detects_shift(self):
        n = 500
        sample = [(i + 0.5) / n for i in range(n)]  # Uniform(0,1)
        shifted_cdf = lambda x: max(0.0, min(1.0, x - 0.3))  # target shifted right
        self.assertGreater(ks_statistic(sample, shifted_cdf), 0.25)

    def test_ks_upper_ci_reproducible(self):
        n = 100
        sample = [(i + 0.5) / n for i in range(n)]
        a = ks_upper_ci(sample, lambda x: x, seed=3)
        b = ks_upper_ci(sample, lambda x: x, seed=3)
        self.assertEqual(a, b)


class TestDiscretizationFloor(unittest.TestCase):
    def test_integer_grid_floor_is_quarter(self):
        # A unit grid -> floor ~ 1/4 (mean abs quantization error over [-0.5, 0.5]).
        self.assertAlmostEqual(discretization_floor(granularity=1.0), 0.25)
        self.assertAlmostEqual(discretization_floor(granularity=0.1), 0.025)

    def test_floor_inferred_from_integer_sample(self):
        sample = [18, 19, 20, 25, 30, 45]  # min positive gap = 1
        self.assertAlmostEqual(discretization_floor(sample), 0.25)

    def test_floor_inferred_from_tenths(self):
        sample = [1.0, 1.1, 1.3, 2.5]  # min positive gap = 0.1
        self.assertAlmostEqual(discretization_floor(sample), 0.025, places=6)

    def test_unknown_granularity_is_zero(self):
        self.assertEqual(discretization_floor([5, 5, 5]), 0.0)  # constant
        self.assertEqual(discretization_floor(None), 0.0)
        self.assertEqual(discretization_floor([]), 0.0)

    def test_is_delta_below_floor(self):
        # Integer field, floor 0.25. slack 0.5 -> flagged only below 0.125.
        below, floor = is_delta_below_floor(0.05, [10, 11, 12, 20])
        self.assertTrue(below)
        self.assertAlmostEqual(floor, 0.25)
        # delta 0.5 is comfortably above the floor -> not flagged
        above, _ = is_delta_below_floor(0.5, [10, 11, 12, 20])
        self.assertFalse(above)
        # a delta between 0.5*floor and floor is a marginal case -> not flagged (slack)
        marginal, _ = is_delta_below_floor(0.2, [10, 11, 12, 20])
        self.assertFalse(marginal)


class TestFamilywiseAlpha(unittest.TestCase):
    def test_bonferroni_splits_budget(self):
        self.assertAlmostEqual(familywise_alpha(0.05, 5), 0.01)

    def test_single_test_unchanged(self):
        self.assertEqual(familywise_alpha(0.05, 1), 0.05)
        self.assertEqual(familywise_alpha(0.05, 0), 0.05)

    def test_none_mode_no_correction(self):
        self.assertEqual(familywise_alpha(0.05, 5, mode="none"), 0.05)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            familywise_alpha(0.05, 5, mode="holm")


if __name__ == "__main__":
    unittest.main()
