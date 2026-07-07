#!/usr/bin/env pytest
"""Unit tests for Mechanism A step 1: the population objective parser, reducers, and
`PopulationValue` attribution. No evaluator wiring is exercised here."""

import re
import statistics
import tempfile
import unittest
from pathlib import Path

from fandango.constraints.population import (
    REDUCER_TARGET_ARITY,
    REDUCERS,
    PopulationValue,
    _correlation,
    _count,
    _distinct_count,
    _exponential_fit,
    _fraction,
    _lognormal_fit,
    _mean,
    _normal_fit,
    _stddev,
    _uniform_fit,
    register_reducer,
    try_parse_population_aggregate,
    wasserstein_fit,
)
from fandango.errors import FandangoValueError
from fandango.evolution.algorithm import DefaultAlgorithm, LoggerLevel
from fandango.language.parse.parse import parse

from .utils import RESOURCES_ROOT

GRAMMAR = """<start> ::= <person>+
<person> ::= <name> "," <age> "\\n"
<name> ::= r'[a-z]+'
<age> ::= r'[0-9]+'
"""


def _parse_spec(objective: str):
    """Parse GRAMMAR + a single objective line, returning (grammar, soft constraint)."""
    spec = GRAMMAR + "\n" + objective + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as f:
        f.write(spec)
        path = f.name
    try:
        with open(path) as f:
            grammar, constraints = parse(f, use_stdlib=False, use_cache=False)
    finally:
        Path(path).unlink(missing_ok=True)
    assert constraints
    return grammar, constraints[0]


def _parse_objective(objective: str):
    return _parse_spec(objective)[1]


def _real_population(grammar, n):
    """A population of real DerivationTrees (needed to satisfy runtime type checks);
    the contents are irrelevant when `_inner_values_per_tree` is stubbed."""
    import random

    random.seed(0)
    return [grammar.fuzz() for _ in range(n)]


class TestReducers(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(_mean([10, 20, 30]), 20)
        self.assertEqual(_mean([]), 0.0)

    def test_stddev(self):
        self.assertEqual(_stddev([]), 0.0)
        self.assertEqual(_stddev([5]), 0.0)  # < 2 values
        self.assertAlmostEqual(_stddev([2, 4, 6]), 1.632993, places=5)

    def test_fraction(self):
        self.assertEqual(_fraction([True, False, True, False]), 0.5)
        self.assertEqual(_fraction([]), 0.0)

    def test_distinct_count(self):
        self.assertEqual(_distinct_count([1, 1, 2, 3, 3]), 3)

    def test_count(self):
        self.assertEqual(_count([1, 1, 2]), 3)

    def test_normal_fit(self):
        # Samples placed exactly on the target quantiles have ~zero Wasserstein distance;
        # a uniform shift moves the distance by (almost) the shift amount.
        nd = statistics.NormalDist(30, 5)
        on_target = [nd.inv_cdf((i + 0.5) / 200) for i in range(200)]
        self.assertAlmostEqual(_normal_fit(on_target, 30, 5), 0.0, places=1)
        shifted = [v + 20 for v in on_target]
        self.assertAlmostEqual(_normal_fit(shifted, 30, 5), 20.0, places=1)
        # Empty population yields no signal (0.0), never a ZeroDivisionError.
        self.assertEqual(_normal_fit([], 30, 5), 0.0)

    def test_normal_fit_rejects_nonpositive_sigma(self):
        with self.assertRaises(FandangoValueError):
            _normal_fit([1, 2, 3], 30, 0)

    def test_distribution_fits_are_zero_on_target(self):
        # Every distributional fit is ~0 when the samples lie on the target's quantiles.
        import math

        n = 300

        def on_quantiles(q):
            return [q((i + 0.5) / n) for i in range(n)]

        nd = statistics.NormalDist(30, 5)
        self.assertAlmostEqual(_normal_fit(on_quantiles(nd.inv_cdf), 30, 5), 0.0, places=1)
        std = statistics.NormalDist(0, 1)
        self.assertAlmostEqual(
            _lognormal_fit(on_quantiles(lambda p: math.exp(std.inv_cdf(p))), 0, 1),
            0.0,
            places=1,
        )
        self.assertAlmostEqual(
            _uniform_fit(on_quantiles(lambda p: 10 + p * 20), 10, 30), 0.0, places=1
        )
        self.assertAlmostEqual(
            _exponential_fit(on_quantiles(lambda p: -math.log1p(-p) / 0.5), 0.5),
            0.0,
            places=1,
        )

    def test_distribution_fits_reject_bad_params(self):
        for call in (
            lambda: _lognormal_fit([1, 2, 3], 0, 0),
            lambda: _uniform_fit([1, 2, 3], 30, 10),  # hi <= lo
            lambda: _exponential_fit([1, 2, 3], 0),
        ):
            with self.assertRaises(FandangoValueError):
                call()

    def test_registry(self):
        self.assertEqual(
            set(REDUCERS),
            {
                "mean",
                "stddev",
                "fraction",
                "distinct_count",
                "count",
                "correlation",
                "normal_fit",
                "lognormal_fit",
                "uniform_fit",
                "exponential_fit",
            },
        )


class TestParsePopulationAggregate(unittest.TestCase):
    def test_accepts_mean(self):
        c = _parse_objective("minimizing abs(mean(int(<age>) for x in population) - 30)")
        agg = try_parse_population_aggregate(c.expression, c.searches)
        self.assertIsNotNone(agg)
        self.assertEqual(agg.reducer_name, "mean")
        self.assertEqual(agg.loop_var, "x")
        # inner has exactly the referenced search; outer replaces the reducer call.
        self.assertEqual(len(agg.inner_searches), 1)
        self.assertIn(next(iter(agg.inner_searches)), agg.inner_expression)
        self.assertIn("___fandango_population_agg___", agg.outer_expression)
        # the generator over `population` is gone from the outer expression.
        self.assertNotIn("for x in population", agg.outer_expression)
        self.assertNotIn("mean", agg.outer_expression)

    def test_accepts_distinct_count(self):
        c = _parse_objective("maximizing distinct_count(<age> for x in population)")
        agg = try_parse_population_aggregate(c.expression, c.searches)
        self.assertIsNotNone(agg)
        self.assertEqual(agg.reducer_name, "distinct_count")

    def test_accepts_listcomp(self):
        c = _parse_objective(
            "minimizing abs(mean([int(<age>) for x in population]) - 30)"
        )
        agg = try_parse_population_aggregate(c.expression, c.searches)
        self.assertIsNotNone(agg)
        self.assertEqual(agg.reducer_name, "mean")

    def test_non_population_returns_none(self):
        c = _parse_objective("minimizing int(<age>)")
        self.assertIsNone(try_parse_population_aggregate(c.expression, c.searches))

    def test_accepts_normal_fit_with_target_params(self):
        # A distributional reducer carries its target as trailing literal args; the
        # generator must be bracketed (list form) once extra args are present.
        c = _parse_objective(
            "minimizing normal_fit([int(<age>) for x in population], 30, 5)"
        )
        agg = try_parse_population_aggregate(c.expression, c.searches)
        self.assertIsNotNone(agg)
        self.assertEqual(agg.reducer_name, "normal_fit")
        self.assertEqual(agg.reducer_args, [30, 5])
        self.assertEqual(len(agg.inner_searches), 1)
        # the outer expression is just the substituted aggregate placeholder.
        self.assertIn("___fandango_population_agg___", agg.outer_expression)
        self.assertNotIn("normal_fit", agg.outer_expression)

    def test_accepts_exponential_fit_single_param(self):
        c = _parse_objective(
            "minimizing exponential_fit([int(<age>) for x in population], 0.5)"
        )
        agg = try_parse_population_aggregate(c.expression, c.searches)
        self.assertEqual(agg.reducer_name, "exponential_fit")
        self.assertEqual(agg.reducer_args, [0.5])

    def test_fit_wrong_arity_raises(self):
        for bad in (
            "normal_fit([int(ph) for x in population], 30)",  # needs 2
            "normal_fit([int(ph) for x in population], 30, 5, 1)",
            "exponential_fit([int(ph) for x in population], 0.5, 1)",  # needs 1
        ):
            with self.assertRaises(FandangoValueError):
                try_parse_population_aggregate(bad, {})

    def test_normal_fit_nonliteral_target_raises(self):
        with self.assertRaises(FandangoValueError):
            try_parse_population_aggregate(
                "normal_fit([int(ph) for x in population], mu, 5)", {}
            )

    # The rejection cases are exercised on crafted (post-substitution) strings so they
    # stay pure unit tests: going through the full parser would raise the same
    # FandangoValueError at convert time (see TestConvertTimeRejection), before we could
    # call try_parse_population_aggregate directly.
    # These reject-paths fire before the searches dict is consulted, so an empty dict
    # is enough (and keeps the runtime type checker happy).
    def test_unknown_reducer_raises(self):
        with self.assertRaises(FandangoValueError):
            try_parse_population_aggregate(
                "median(int(ph) for x in population)", {}
            )

    def test_multiple_aggregates_raise(self):
        with self.assertRaises(FandangoValueError):
            try_parse_population_aggregate(
                "mean(a for x in population) - count(b for x in population)", {}
            )

    def test_filter_raises(self):
        with self.assertRaises(FandangoValueError):
            try_parse_population_aggregate(
                "mean(int(ph) for x in population if ph)", {}
            )

    def test_population_outside_reducer_raises(self):
        with self.assertRaises(FandangoValueError):
            try_parse_population_aggregate("population + 1", {})


class TestConvertTimeRejection(unittest.TestCase):
    """Malformed population objectives surface at parse/convert time."""

    def test_unknown_reducer(self):
        with self.assertRaises(FandangoValueError):
            _parse_objective("minimizing median(int(<age>) for x in population)")

    def test_population_outside_reducer(self):
        with self.assertRaises(FandangoValueError):
            _parse_objective("minimizing population + 1")


class TestRegisterReducer(unittest.TestCase):
    """The downstream extension point: register a custom reducer/distribution by name."""

    def tearDown(self):
        # Keep the process-wide registry clean for other tests.
        REDUCERS.pop("triangular_fit", None)
        REDUCER_TARGET_ARITY.pop("triangular_fit", None)

    def test_register_and_use_custom_fit(self):
        # A symmetric triangular distribution on [lo, hi] via its closed-form quantile,
        # built on the public wasserstein_fit helper — the intended extension recipe.
        def _tri_quantile(p, lo, hi):
            mid = (lo + hi) / 2
            if p < 0.5:
                return lo + ((hi - lo) * (mid - lo) * p) ** 0.5
            return hi - ((hi - lo) * (hi - mid) * (1 - p)) ** 0.5

        register_reducer(
            "triangular_fit",
            lambda values, lo, hi: wasserstein_fit(values, lambda p: _tri_quantile(p, lo, hi)),
            target_arity=2,
        )
        self.assertIn("triangular_fit", REDUCERS)
        self.assertEqual(REDUCER_TARGET_ARITY["triangular_fit"], 2)

        # It now parses like any built-in fit, capturing its two target params.
        c = _parse_objective(
            "minimizing triangular_fit([int(<age>) for x in population], 0, 100)"
        )
        agg = try_parse_population_aggregate(c.expression, c.searches)
        self.assertEqual(agg.reducer_name, "triangular_fit")
        self.assertEqual(agg.reducer_args, [0, 100])
        # On-target samples give ~0 distance.
        on_target = [_tri_quantile((i + 0.5) / 300, 0, 100) for i in range(300)]
        self.assertAlmostEqual(
            REDUCERS["triangular_fit"](on_target, 0, 100), 0.0, places=1
        )

    def test_zero_arity_reducer_takes_no_target(self):
        register_reducer("triangular_fit", lambda values: float(len(values)))
        self.assertNotIn("triangular_fit", REDUCER_TARGET_ARITY)
        with self.assertRaises(FandangoValueError):
            # supplying a target param to a 0-arity reducer is an arity error
            try_parse_population_aggregate(
                "triangular_fit([int(ph) for x in population], 5)", {}
            )

    def test_invalid_name_or_arity_rejected(self):
        with self.assertRaises(FandangoValueError):
            register_reducer("not an identifier", lambda values: 0.0)
        with self.assertRaises(FandangoValueError):
            register_reducer("triangular_fit", lambda values: 0.0, target_arity=-1)


JOINT_GRAMMAR = """<start> ::= <person> "\\n" <person> "\\n" <person>
<person> ::= <age> "," <income>
<age> ::= <digit> <digit>
<income> ::= <digit> <digit>
<digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
"""


def _parse_joint(objective):
    spec = JOINT_GRAMMAR + "\n" + objective + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as f:
        f.write(spec)
        path = f.name
    try:
        with open(path) as f:
            grammar, constraints = parse(f, use_stdlib=False, use_cache=False)
    finally:
        Path(path).unlink(missing_ok=True)
    return grammar, constraints[0]


class TestJointRowScoping(unittest.TestCase):
    """Prototype: an objective combining >= 2 fields is paired *per row*, not
    cross-producted over the whole tree (which would destroy every joint statistic)."""

    def _pv(self, objective):
        grammar, c = _parse_joint(objective)
        agg = try_parse_population_aggregate(c.expression, c.searches)
        pv = PopulationValue(
            c.optimization_goal,
            c.expression,
            aggregate=agg,
            local_variables=c.local_variables,
            global_variables=c.global_variables,
        )
        return grammar, pv

    def test_marginal_stays_tree_scoped(self):
        _, pv = self._pv("minimizing abs(mean(int(<age>) for x in population) - 30)")
        self.assertFalse(pv._row_scoped)

    def test_joint_pairs_row_wise_not_cross_product(self):
        grammar, pv = self._pv(
            "maximizing correlation((int(<age>), int(<income>)) for x in population)"
        )
        self.assertTrue(pv._row_scoped)
        self.assertEqual(pv._target_symbols, ["<age>", "<income>"])
        tree = grammar.parse("10,90\n20,80\n30,70")
        per_tree = pv._inner_values_per_tree([tree])
        # The diagonal (3 aligned pairs), NOT the 9-element cross product.
        self.assertEqual(per_tree, [[(10, 90), (20, 80), (30, 70)]])
        self.assertEqual(str(pv._row_symbol), "<person>")

    def test_correlation_recovered_not_destroyed(self):
        grammar, pv = self._pv(
            "maximizing correlation((int(<age>), int(<income>)) for x in population)"
        )
        tree = grammar.parse("10,90\n20,80\n30,70")  # perfectly negative row-wise
        vals = [v for vs in pv._inner_values_per_tree([tree]) for v in vs]
        # Row-wise this is -1.0; a cross product would report ~0.0.
        self.assertAlmostEqual(pv._outer_score(vals), -1.0, places=6)

    def test_correlation_reducer_rejects_non_pairs(self):
        with self.assertRaises(FandangoValueError):
            _correlation([1, 2, 3])
        self.assertEqual(_correlation([(1, 2)]), 0.0)  # < 2 pairs -> undefined -> 0.0


def _population_value(objective, attribution):
    grammar, c = _parse_spec(objective)
    agg = try_parse_population_aggregate(c.expression, c.searches)
    pv = PopulationValue(
        c.optimization_goal,
        c.expression,
        aggregate=agg,
        attribution=attribution,
        local_variables=c.local_variables,
        global_variables=c.global_variables,
    )
    return pv, grammar


class TestPopulationValueAttribution(unittest.TestCase):
    OBJECTIVE = "minimizing abs(mean(int(<age>) for x in population) - 30)"

    def test_bad_attribution_raises(self):
        with self.assertRaises(FandangoValueError):
            _population_value(self.OBJECTIVE, "nonsense")

    def test_neutral_per_tree_fitness(self):
        pv, grammar = _population_value(self.OBJECTIVE, "loo")
        self.assertEqual(pv.fitness(grammar.fuzz()).values, [])

    def test_empty_population(self):
        pv, _ = _population_value(self.OBJECTIVE, "loo")
        self.assertEqual(pv.evaluate_population([]), [])

    def test_uniform_is_flat(self):
        pv, grammar = _population_value(self.OBJECTIVE, "uniform")
        # stub the per-tree inner values so the test is deterministic.
        pv._inner_values_per_tree = lambda pop: [[30], [10], [90], [50]]
        bonuses = pv.evaluate_population(_real_population(grammar, 4))
        self.assertEqual(len(bonuses), 4)
        self.assertEqual(len(set(bonuses)), 1)  # every individual identical

    def test_loo_rewards_the_helper(self):
        # mean target is 30; individual 0 sits exactly on target, the rest are far off,
        # so removing individual 0 pushes the mean furthest from 30 -> highest reward.
        pv, grammar = _population_value(self.OBJECTIVE, "loo")
        pv._inner_values_per_tree = lambda pop: [[30], [100], [100], [100]]
        bonuses = pv.evaluate_population(_real_population(grammar, 4))
        self.assertEqual(len(bonuses), 4)
        self.assertEqual(bonuses[0], max(bonuses))
        self.assertGreater(bonuses[0], bonuses[1])
        # the three equally-unhelpful individuals tie.
        self.assertEqual(bonuses[1], bonuses[2])
        self.assertEqual(bonuses[2], bonuses[3])

    def test_loo_no_inner_values_returns_zeros(self):
        pv, grammar = _population_value(self.OBJECTIVE, "loo")
        pv._inner_values_per_tree = lambda pop: [[], [], []]
        self.assertEqual(
            pv.evaluate_population(_real_population(grammar, 3)), [0.0, 0.0, 0.0]
        )

    def test_maximize_distinct_count_ranks_unique_contributor(self):
        pv, grammar = _population_value(
            "maximizing distinct_count(int(<age>) for x in population)", "loo"
        )
        # individuals 1..3 are duplicates; individual 0 is the only unique value, so
        # removing it drops the distinct count the most -> it is the top contributor.
        pv._inner_values_per_tree = lambda pop: [[7], [5], [5], [5]]
        bonuses = pv.evaluate_population(_real_population(grammar, 4))
        self.assertEqual(bonuses[0], max(bonuses))


class TestPopulationEndToEnd(unittest.TestCase):
    """End-to-end: the objective must steer the population distribution relative to the
    same grammar run *without* it. These are soft, best-effort objectives, so we assert
    a clear directional shift rather than exact convergence."""

    GENERATIONS = 100
    POPULATION_SIZE = 40
    SEED = 1

    def _final_population(self, spec_file, *, with_objective):
        with open(spec_file) as f:
            grammar, constraints = parse(f, use_stdlib=False, use_cache=False)
        fandango = DefaultAlgorithm(
            grammar=grammar,
            constraints=constraints if with_objective else [],
            random_seed=self.SEED,
            logger_level=LoggerLevel.ERROR,
            population_size=self.POPULATION_SIZE,
        )
        # Drain the generator so the GA runs the full generation budget; we inspect the
        # final working set rather than the (immediately-satisfied) solution stream.
        list(fandango.generate(max_generations=self.GENERATIONS))
        return fandango.population

    @staticmethod
    def _numbers(trees):
        values = []
        for tree in trees:
            values.extend(int(m) for m in re.findall(r"\d\d", str(tree)))
        return values

    def test_mean_converges_toward_target(self):
        target = 30
        base = self._numbers(
            self._final_population(
                RESOURCES_ROOT / "population_mean.fan", with_objective=False
            )
        )
        opt = self._numbers(
            self._final_population(
                RESOURCES_ROOT / "population_mean.fan", with_objective=True
            )
        )
        base_mean, opt_mean = statistics.mean(base), statistics.mean(opt)
        # The objective should pull the mean clearly closer to the target than baseline.
        self.assertLess(
            abs(opt_mean - target),
            abs(base_mean - target) - 2.0,
            f"objective mean {opt_mean:.1f} not meaningfully closer to {target} "
            f"than baseline {base_mean:.1f}",
        )

    def test_maximizing_pushes_mean_up(self):
        # Exercises the "max" goal path: the mean should end up clearly above the
        # no-objective baseline (which hovers around the uniform midpoint ~49.5).
        base = statistics.mean(
            self._numbers(
                self._final_population(
                    RESOURCES_ROOT / "population_maximize.fan", with_objective=False
                )
            )
        )
        opt = statistics.mean(
            self._numbers(
                self._final_population(
                    RESOURCES_ROOT / "population_maximize.fan", with_objective=True
                )
            )
        )
        self.assertGreater(
            opt,
            base + 4.0,
            f"maximizing mean {opt:.1f} not clearly above baseline {base:.1f}",
        )


class TestAttributionKnob(unittest.TestCase):
    """The run-level population_attribution override (constructor + CLI plumbing)."""

    OBJECTIVE = "minimizing abs(mean(int(<age>) for x in population) - 30)"

    def _grammar_and_constraints(self):
        spec = GRAMMAR + "\n" + self.OBJECTIVE + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as f:
            f.write(spec)
            path = f.name
        try:
            with open(path) as f:
                grammar, constraints = parse(f, use_stdlib=False, use_cache=False)
        finally:
            Path(path).unlink(missing_ok=True)
        return grammar, constraints

    def test_evaluator_applies_override(self):
        from fandango.evolution.evaluation import Evaluator

        grammar, constraints = self._grammar_and_constraints()
        # objectives default to "loo" at convert time...
        self.assertEqual(constraints[0].attribution, "loo")
        # ...and the run-level knob overrides them when the evaluator is built.
        evaluator = Evaluator(
            grammar,
            constraints,
            expected_fitness=1.0,
            diversity_k=0,
            diversity_weight=0.0,
            population_attribution="uniform",
        )
        self.assertEqual(len(evaluator._population_constraints), 1)
        self.assertEqual(evaluator._population_constraints[0].attribution, "uniform")

    def test_evaluator_rejects_bad_override(self):
        from fandango.evolution.evaluation import Evaluator

        grammar, constraints = self._grammar_and_constraints()
        with self.assertRaises(ValueError):
            Evaluator(
                grammar,
                constraints,
                expected_fitness=1.0,
                diversity_k=0,
                diversity_weight=0.0,
                population_attribution="nonsense",
            )

    def test_cli_flag_flows_into_settings(self):
        from fandango.cli.parser import get_parser
        from fandango.cli.utils import make_fandango_settings

        parser = get_parser()
        args = parser.parse_args(
            ["fuzz", "-f", str(RESOURCES_ROOT / "population_mean.fan"),
             "--population-attribution", "uniform"]
        )
        settings = make_fandango_settings(args)
        self.assertEqual(settings.get("population_attribution"), "uniform")

    def test_cli_flag_absent_by_default(self):
        from fandango.cli.parser import get_parser
        from fandango.cli.utils import make_fandango_settings

        parser = get_parser()
        args = parser.parse_args(
            ["fuzz", "-f", str(RESOURCES_ROOT / "population_mean.fan")]
        )
        settings = make_fandango_settings(args)
        # Not set -> the constructor default ("loo") applies.
        self.assertNotIn("population_attribution", settings)


if __name__ == "__main__":
    unittest.main()
