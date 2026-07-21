#!/usr/bin/env pytest
"""Mechanism B steps 5-6: the population sampler that *constructs* a batch satisfying a hard
population `where`. v1 covers the `fraction` quota, `distinct_count` diversity, and distributional
fits (normal_fit &c.); other shapes raise clear errors."""

import random
import tempfile
import unittest
from pathlib import Path

from fandango import Fandango
from fandango.constraints.population import REDUCERS
from fandango.constraints.population_sampler import (
    PopulationSampler,
    PopulationShortfallError,
)
from fandango.errors import FandangoValueError
from fandango.language.parse.parse import parse

# One record per tree: N trees == N individuals, so the fraction is over the batch directly.
GRAMMAR = """<start> ::= <income> "\\n"
<income> ::= "0" | "1"
"""


def _grammar_with(*where_lines: str):
    spec = GRAMMAR + "\n" + "\n".join(where_lines) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as f:
        f.write(spec)
        path = f.name
    try:
        with open(path) as f:
            grammar, _ = parse(f, use_stdlib=False, use_cache=False)
        return grammar
    finally:
        Path(path).unlink(missing_ok=True)


def _income_fraction(batch):
    """The fraction of a batch of `<income>` records whose income == 1."""
    return REDUCERS["fraction"]([int(str(t).strip()) == 1 for t in batch])


class TestPopulationSamplerQuota(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_exact_fraction_is_guaranteed(self):
        grammar = _grammar_with(
            "where fraction(int(<income>) == 1 for x in population) == 0.30"
        )
        batch = PopulationSampler(grammar).sample(20)
        self.assertEqual(len(batch), 20)
        # Exactly 6 of 20 satisfy the predicate -> fraction is precisely 0.30, by construction.
        self.assertEqual(sum(int(str(t).strip()) == 1 for t in batch), 6)
        self.assertEqual(_income_fraction(batch), 0.30)

    def test_greater_equal_meets_boundary(self):
        grammar = _grammar_with(
            "where fraction(int(<income>) == 1 for x in population) >= 0.30"
        )
        batch = PopulationSampler(grammar).sample(20)
        self.assertGreaterEqual(_income_fraction(batch), 0.30)
        # `>=` takes the minimal satisfying count: ceil(0.30 * 20) = 6.
        self.assertEqual(sum(int(str(t).strip()) == 1 for t in batch), 6)

    def test_less_equal_meets_boundary(self):
        grammar = _grammar_with(
            "where fraction(int(<income>) == 1 for x in population) <= 0.25"
        )
        batch = PopulationSampler(grammar).sample(20)
        self.assertLessEqual(_income_fraction(batch), 0.25)
        self.assertEqual(sum(int(str(t).strip()) == 1 for t in batch), 5)  # floor(0.25*20)

    def test_exact_fraction_snaps_and_warns_at_awkward_n(self):
        grammar = _grammar_with(
            "where fraction(int(<income>) == 1 for x in population) == 0.30"
        )
        # 0.30 * 7 = 2.1 -> snaps to 2/7; the sampler warns about the effective target.
        with self.assertLogs("fandango", level="WARNING") as cm:
            batch = PopulationSampler(grammar).sample(7)
        self.assertEqual(sum(int(str(t).strip()) == 1 for t in batch), 2)
        self.assertTrue(any("not achievable at N=7" in m for m in cm.output))

    def test_fraction_out_of_range_is_rejected(self):
        grammar = _grammar_with(
            "where fraction(int(<income>) == 1 for x in population) == 1.5"
        )
        with self.assertRaises(FandangoValueError):
            PopulationSampler(grammar).sample(10)

    def test_no_requirements_returns_plain_batch(self):
        grammar = _grammar_with()  # no population `where`
        batch = PopulationSampler(grammar).sample(5)
        self.assertEqual(len(batch), 5)


class TestPopulationSamplerHonestLimits(unittest.TestCase):
    """v1 refuses -- clearly -- everything outside the fraction-quota slice."""

    def setUp(self):
        random.seed(0)

    def test_overlapping_requirements_raise(self):
        # Two requirements on the SAME field (<income>) cannot be jointly constructed in v1.
        grammar = _grammar_with(
            "where fraction(int(<income>) == 1 for x in population) == 0.30",
            "where fraction(int(<income>) == 0 for x in population) >= 0.10",
        )
        with self.assertRaises(FandangoValueError):
            PopulationSampler(grammar).sample(10)

    def test_unsupported_reducer_raises(self):
        # fraction (quota) and distinct_count (diversity) are constructed; mean is not (yet).
        grammar = _grammar_with(
            "where mean(int(<income>) for x in population) <= 1"
        )
        with self.assertRaises(NotImplementedError):
            PopulationSampler(grammar).sample(10)

    def test_multi_valued_field_raises(self):
        # A tree that yields several incomes is the pooled/grouping case, unsupported in v1.
        spec = '<start> ::= <income>+ "\\n"\n<income> ::= "0" | "1"\n'
        spec += "where fraction(int(<income>) == 1 for x in population) == 0.30\n"
        with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as f:
            f.write(spec)
            path = f.name
        try:
            with open(path) as f:
                grammar, _ = parse(f, use_stdlib=False, use_cache=False)
        finally:
            Path(path).unlink(missing_ok=True)
        with self.assertRaises(NotImplementedError):
            PopulationSampler(grammar).sample(10)


OCC_GRAMMAR = (
    '<start> ::= <occ> "\\n"\n<occ> ::= "eng" | "doc" | "art" | "law" | "edu"\n'  # 5 values
)


def _occ_grammar_with(where_line: str):
    spec = OCC_GRAMMAR + "\n" + where_line + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as f:
        f.write(spec)
        path = f.name
    try:
        with open(path) as f:
            grammar, _ = parse(f, use_stdlib=False, use_cache=False)
        return grammar
    finally:
        Path(path).unlink(missing_ok=True)


def _distinct(batch):
    return len({str(t).strip() for t in batch})


class TestPopulationSamplerDiversity(unittest.TestCase):
    """distinct_count(<field>) OP K: build a batch with the required number of distinct values."""

    def setUp(self):
        random.seed(0)

    def test_at_least_builds_target_distinct(self):
        batch = PopulationSampler(
            _occ_grammar_with("where distinct_count(<occ> for x in population) >= 3")
        ).sample(20)
        self.assertEqual(len(batch), 20)
        self.assertGreaterEqual(_distinct(batch), 3)

    def test_exact_distinct(self):
        batch = PopulationSampler(
            _occ_grammar_with("where distinct_count(<occ> for x in population) == 4")
        ).sample(20)
        self.assertEqual(_distinct(batch), 4)

    def test_greater_than_boundary(self):
        batch = PopulationSampler(
            _occ_grammar_with("where distinct_count(<occ> for x in population) > 3")
        ).sample(20)
        self.assertGreater(_distinct(batch), 3)

    def test_at_most_caps_distinct(self):
        batch = PopulationSampler(
            _occ_grammar_with("where distinct_count(<occ> for x in population) <= 2")
        ).sample(20)
        self.assertEqual(len(batch), 20)
        self.assertLessEqual(_distinct(batch), 2)

    def test_less_than_boundary(self):
        batch = PopulationSampler(
            _occ_grammar_with("where distinct_count(<occ> for x in population) < 3")
        ).sample(20)
        self.assertLess(_distinct(batch), 3)

    def test_more_distinct_than_grammar_can_produce_shortfalls(self):
        # The grammar has only 5 distinct occupations; 6 is unreachable.
        grammar = _occ_grammar_with(
            "where distinct_count(<occ> for x in population) >= 6"
        )
        with self.assertRaises(PopulationShortfallError):
            PopulationSampler(grammar, max_attempts_per_slot=50).sample(20)

    def test_more_distinct_than_batch_size_is_infeasible(self):
        grammar = _occ_grammar_with(
            "where distinct_count(<occ> for x in population) >= 4"
        )
        with self.assertRaises(FandangoValueError):
            PopulationSampler(grammar).sample(3)

    def test_non_integer_exact_target_is_rejected(self):
        grammar = _occ_grammar_with(
            "where distinct_count(<occ> for x in population) == 2.5"
        )
        with self.assertRaises(FandangoValueError):
            PopulationSampler(grammar).sample(20)


AGE_GRAMMAR = '<start> ::= <name> "," <age> "\\n"\n<name> ::= "x" | "y"\n<age> ::= r\'[0-9][0-9]\'\n'


def _age_grammar_with(where_line: str, age_body: str = "r'[0-9][0-9]'"):
    grammar_src = (
        f'<start> ::= <name> "," <age> "\\n"\n<name> ::= "x" | "y"\n<age> ::= {age_body}\n'
    )
    spec = grammar_src + "\n" + where_line + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as f:
        f.write(spec)
        path = f.name
    try:
        with open(path) as f:
            grammar, _ = parse(f, use_stdlib=False, use_cache=False)
        return grammar
    finally:
        Path(path).unlink(missing_ok=True)


def _ages(batch):
    return [int(str(t).strip().split(",")[1]) for t in batch]


class TestPopulationSamplerDistribution(unittest.TestCase):
    """A distributional fit `reducer([<x> for x in population], ...) <= delta` is matched by
    fuzzing a pool and selecting the individual nearest each target quantile."""

    def setUp(self):
        random.seed(0)

    def test_normal_fit_matches_within_delta(self):
        grammar = _age_grammar_with(
            "where normal_fit([int(<age>) for x in population], 30, 5) <= 0.5"
        )
        batch = PopulationSampler(grammar).sample(30)
        self.assertEqual(len(batch), 30)
        self.assertLessEqual(REDUCERS["normal_fit"](_ages(batch), 30, 5), 0.5)

    def test_uniform_fit_matches_within_delta(self):
        grammar = _age_grammar_with(
            "where uniform_fit([int(<age>) for x in population], 10, 40) <= 0.5"
        )
        batch = PopulationSampler(grammar).sample(40)
        self.assertLessEqual(REDUCERS["uniform_fit"](_ages(batch), 10, 40), 0.5)

    def test_coarse_grammar_shortfalls(self):
        # Only five possible ages can't approximate Normal(30, 5) to within 0.5.
        grammar = _age_grammar_with(
            "where normal_fit([int(<age>) for x in population], 30, 5) <= 0.5",
            age_body='"18" | "25" | "30" | "35" | "45"',
        )
        with self.assertRaises(PopulationShortfallError):
            PopulationSampler(grammar).sample(30)

    def test_fit_with_lower_bound_operator_is_rejected(self):
        # A fit is a distance to a target; `>=` ("stay far away") has no construction meaning.
        grammar = _age_grammar_with(
            "where normal_fit([int(<age>) for x in population], 30, 5) >= 0.5"
        )
        with self.assertRaises(NotImplementedError):
            PopulationSampler(grammar).sample(30)

    def test_delta_below_discretization_floor_is_diagnosed(self):
        # Integer ages step by 1 -> discretization floor ~0.25. A delta of 0.05 is
        # unsatisfiable *in principle* (no placement of a rounded field gets that close to
        # a continuous Normal), and must be diagnosed precisely, not as a generic shortfall.
        grammar = _age_grammar_with(
            "where normal_fit([int(<age>) for x in population], 30, 5) <= 0.05"
        )
        with self.assertRaises(PopulationShortfallError) as ctx:
            PopulationSampler(grammar).sample(30)
        message = str(ctx.exception)
        self.assertIn("discretization floor", message)
        self.assertIn("unsatisfiable in principle", message)


class TestPopulationSamplerCoEnforcement(unittest.TestCase):
    """A per-tree hard `where` alongside a population requirement: every constructed individual
    satisfies the per-tree constraint (rejection-fuzzed candidate source), and the population
    requirement still holds."""

    # A flag whose fraction is constrained at the batch level, and an independent age with a
    # per-tree lower bound -- so the two constraints don't interact.
    SPEC = (
        '<start> ::= <flag> "," <age> "\\n"\n'
        '<flag> ::= "0" | "1"\n'
        "<age> ::= r'[0-9][0-9]'\n"
        "where int(<age>) >= 40\n"
        "where fraction(int(<flag>) == 1 for x in population) == 0.30\n"
    )

    def setUp(self):
        random.seed(0)

    @staticmethod
    def _parse(spec):
        with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as f:
            f.write(spec)
            path = f.name
        try:
            with open(path) as f:
                return parse(f, use_stdlib=False, use_cache=False)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_sampler_co_enforces_per_tree_constraint(self):
        grammar, constraints = self._parse(self.SPEC)
        batch = PopulationSampler(grammar, constraints=constraints).sample(20)
        self.assertEqual(len(batch), 20)
        ages = [int(str(t).strip().split(",")[1]) for t in batch]
        flags = [int(str(t).strip().split(",")[0]) for t in batch]
        self.assertTrue(all(a >= 40 for a in ages))  # per-tree constraint
        self.assertEqual(REDUCERS["fraction"]([f == 1 for f in flags]), 0.30)  # population

    def test_unsatisfiable_per_tree_constraint_shortfalls(self):
        # <age> is two digits (0-99); demanding >= 200 can never be fuzzed.
        grammar, constraints = self._parse(
            '<start> ::= <age> "\\n"\n'
            "<age> ::= r'[0-9][0-9]'\n"
            "where int(<age>) >= 200\n"
            "where fraction(int(<age>) >= 50 for x in population) == 0.30\n"
        )
        with self.assertRaises(PopulationShortfallError):
            PopulationSampler(
                grammar, constraints=constraints, max_attempts_per_slot=50
            ).sample(10)


# Flat record: three sibling fields, so requirements on them are structurally disjoint.
JOINT_GRAMMAR = (
    '<start> ::= <income> "," <occ> "," <age> "\\n"\n'
    '<income> ::= "0" | "1"\n'
    '<occ> ::= "eng" | "doc" | "art" | "law" | "edu"\n'
    "<age> ::= r'[0-9][0-9]'\n"
)


def _joint_grammar(*where_lines, grammar_src=JOINT_GRAMMAR):
    spec = grammar_src + "\n" + "\n".join(where_lines) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as f:
        f.write(spec)
        path = f.name
    try:
        with open(path) as f:
            return parse(f, use_stdlib=False, use_cache=False)
    finally:
        Path(path).unlink(missing_ok=True)


def _fields(batch):
    """(income, occ, age) columns from a batch of `<income>,<occ>,<age>` records."""
    rows = [str(t).strip().split(",") for t in batch]
    return (
        [int(r[0]) for r in rows],
        [r[1] for r in rows],
        [int(r[2]) for r in rows],
    )


class TestPopulationSamplerJoint(unittest.TestCase):
    """Multiple population requirements on disjoint fields, constructed together by grafting."""

    def setUp(self):
        random.seed(0)

    def test_three_disjoint_requirements_all_hold(self):
        grammar, constraints = _joint_grammar(
            "where fraction(int(<income>) == 1 for x in population) == 0.30",
            "where distinct_count(<occ> for x in population) >= 3",
            "where normal_fit([int(<age>) for x in population], 30, 5) <= 0.6",
        )
        batch = PopulationSampler(grammar, constraints=constraints).sample(30)
        incomes, occs, ages = _fields(batch)
        self.assertEqual(len(batch), 30)
        self.assertEqual(REDUCERS["fraction"]([i == 1 for i in incomes]), 0.30)
        self.assertGreaterEqual(REDUCERS["distinct_count"](occs), 3)
        self.assertLessEqual(REDUCERS["normal_fit"](ages, 30, 5), 0.6)

    def test_two_disjoint_fractions(self):
        grammar, constraints = _joint_grammar(
            "where fraction(int(<income>) == 1 for x in population) == 0.30",
            "where fraction(int(<age>) >= 50 for x in population) == 0.50",
        )
        batch = PopulationSampler(grammar, constraints=constraints).sample(20)
        incomes, _, ages = _fields(batch)
        self.assertEqual(REDUCERS["fraction"]([i == 1 for i in incomes]), 0.30)
        self.assertEqual(REDUCERS["fraction"]([a >= 50 for a in ages]), 0.50)

    def test_overlapping_fields_rejected(self):
        grammar, constraints = _joint_grammar(
            "where distinct_count(<occ> for x in population) >= 3",
            "where fraction(<occ> == 'eng' for x in population) == 0.20",
        )
        with self.assertRaises(FandangoValueError):
            PopulationSampler(grammar, constraints=constraints).sample(20)

    def test_multi_symbol_requirement_rejected(self):
        # Inner reads two symbols (<income> and <age>) -> row-scoped/coupled, out of scope.
        grammar, constraints = _joint_grammar(
            "where fraction((int(<income>) == 1 and int(<age>) >= 30) for x in population) == 0.30",
        )
        with self.assertRaises(NotImplementedError):
            PopulationSampler(grammar, constraints=constraints).sample(20)

    def test_nested_fields_rejected(self):
        # <age> derives under <person>; distinct symbols but their grafts would collide.
        src = (
            '<start> ::= <person> "\\n"\n'
            '<person> ::= <name> "," <age>\n'
            '<name> ::= "x" | "y" | "z"\n'
            "<age> ::= r'[0-9][0-9]'\n"
        )
        grammar, constraints = _joint_grammar(
            "where distinct_count(<person> for x in population) >= 2",
            "where fraction(int(<age>) >= 50 for x in population) == 0.50",
            grammar_src=src,
        )
        with self.assertRaises(FandangoValueError):
            PopulationSampler(grammar, constraints=constraints).sample(10)

    def test_joint_infeasible_subrequirement_raises(self):
        # distinct_count >= 25 at N=20 is infeasible (can't hold 25 distinct in 20 individuals).
        grammar, constraints = _joint_grammar(
            "where fraction(int(<income>) == 1 for x in population) == 0.30",
            "where distinct_count(<occ> for x in population) >= 25",
        )
        with self.assertRaises(FandangoValueError):
            PopulationSampler(grammar, constraints=constraints).sample(20)


CORR_GRAMMAR = (
    '<start> ::= <age> "," <income> "\\n"\n'
    "<age> ::= r'[0-9][0-9]'\n"
    "<income> ::= r'[0-9][0-9][0-9]'\n"
)


def _corr_grammar(*where_lines, grammar_src=CORR_GRAMMAR):
    spec = grammar_src + "\n" + "\n".join(where_lines) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as f:
        f.write(spec)
        path = f.name
    try:
        with open(path) as f:
            return parse(f, use_stdlib=False, use_cache=False)
    finally:
        Path(path).unlink(missing_ok=True)


def _age_income_pairs(batch):
    rows = [str(t).strip().split(",") for t in batch]
    return [(int(r[0]), int(r[1])) for r in rows]


class TestPopulationSamplerCorrelation(unittest.TestCase):
    """Coupled `correlation((<x>, <y>)) OP r`: pair two fields so their per-individual correlation
    meets the bound, then graft both together."""

    def setUp(self):
        random.seed(0)

    def test_positive_correlation_achieved(self):
        grammar, constraints = _corr_grammar(
            "where correlation((int(<age>), int(<income>)) for x in population) >= 0.5"
        )
        batch = PopulationSampler(grammar, constraints=constraints).sample(30)
        self.assertEqual(len(batch), 30)
        self.assertGreaterEqual(REDUCERS["correlation"](_age_income_pairs(batch)), 0.5)

    def test_negative_correlation_achieved(self):
        grammar, constraints = _corr_grammar(
            "where correlation((int(<age>), int(<income>)) for x in population) <= -0.5"
        )
        batch = PopulationSampler(grammar, constraints=constraints).sample(30)
        self.assertLessEqual(REDUCERS["correlation"](_age_income_pairs(batch)), -0.5)

    def test_reversed_tuple_order(self):
        # (income, age) -- position order must drive which field is grafted, not sorted order.
        grammar, constraints = _corr_grammar(
            "where correlation((int(<income>), int(<age>)) for x in population) >= 0.5"
        )
        batch = PopulationSampler(grammar, constraints=constraints).sample(30)
        self.assertGreaterEqual(REDUCERS["correlation"](_age_income_pairs(batch)), 0.5)

    def test_coupled_plus_disjoint_single_field(self):
        src = (
            '<start> ::= <age> "," <income> "," <occ> "\\n"\n'
            "<age> ::= r'[0-9][0-9]'\n"
            "<income> ::= r'[0-9][0-9][0-9]'\n"
            '<occ> ::= "eng" | "doc" | "art" | "law" | "edu"\n'
        )
        grammar, constraints = _corr_grammar(
            "where correlation((int(<age>), int(<income>)) for x in population) >= 0.5",
            "where distinct_count(<occ> for x in population) >= 3",
            grammar_src=src,
        )
        batch = PopulationSampler(grammar, constraints=constraints).sample(30)
        rows = [str(t).strip().split(",") for t in batch]
        pairs = [(int(r[0]), int(r[1])) for r in rows]
        self.assertGreaterEqual(REDUCERS["correlation"](pairs), 0.5)
        self.assertGreaterEqual(REDUCERS["distinct_count"]([r[2] for r in rows]), 3)

    def test_exact_correlation_rejected(self):
        grammar, constraints = _corr_grammar(
            "where correlation((int(<age>), int(<income>)) for x in population) == 0.5"
        )
        with self.assertRaises(NotImplementedError):
            PopulationSampler(grammar, constraints=constraints).sample(30)

    def test_wrong_symbol_count_rejected(self):
        grammar, constraints = _corr_grammar(
            "where correlation(int(<age>) for x in population) >= 0.5"
        )
        with self.assertRaises(NotImplementedError):
            PopulationSampler(grammar, constraints=constraints).sample(30)

    def test_overlapping_field_rejected(self):
        grammar, constraints = _corr_grammar(
            "where correlation((int(<age>), int(<income>)) for x in population) >= 0.5",
            "where fraction(int(<age>) >= 50 for x in population) == 0.30",
        )
        with self.assertRaises(FandangoValueError):
            PopulationSampler(grammar, constraints=constraints).sample(30)

    def test_shortfall_when_unreachable(self):
        # A constant income column has zero variance -> correlation is 0.0, never >= 0.5.
        grammar, constraints = _corr_grammar(
            "where correlation((int(<age>), int(<income>)) for x in population) >= 0.5",
            grammar_src='<start> ::= <age> "," <income> "\\n"\n'
            "<age> ::= r'[0-9][0-9]'\n"
            '<income> ::= "5"\n',
        )
        with self.assertRaises(PopulationShortfallError):
            PopulationSampler(grammar, constraints=constraints).sample(30)


class TestPopulationSamplerEndToEnd(unittest.TestCase):
    """Through the public `Fandango.fuzz` API: a population `where` routes to the sampler."""

    SPEC = (
        '<start> ::= <income> "\\n"\n'
        '<income> ::= "0" | "1"\n'
        "where fraction(int(<income>) == 1 for x in population) == 0.30\n"
    )

    OCC_SPEC = (
        '<start> ::= <occ> "\\n"\n'
        '<occ> ::= "eng" | "doc" | "art" | "law" | "edu"\n'
        "where distinct_count(<occ> for x in population) >= 4\n"
    )

    AGE_SPEC = (
        '<start> ::= <age> "\\n"\n'
        "<age> ::= r'[0-9][0-9]'\n"
        "where normal_fit([int(<age>) for x in population], 30, 5) <= 0.5\n"
    )

    def setUp(self):
        random.seed(0)

    def test_fuzz_constructs_the_guaranteed_batch(self):
        fan = Fandango(self.SPEC)
        batch = fan.fuzz(desired_solutions=20)
        self.assertEqual(len(batch), 20)
        self.assertEqual(_income_fraction(batch), 0.30)

    def test_fuzz_without_batch_size_raises(self):
        fan = Fandango(self.SPEC)
        with self.assertRaises(FandangoValueError):
            fan.fuzz()

    def test_fuzz_constructs_the_diverse_batch(self):
        fan = Fandango(self.OCC_SPEC)
        batch = fan.fuzz(desired_solutions=20)
        self.assertEqual(len(batch), 20)
        self.assertGreaterEqual(_distinct(batch), 4)

    def test_fuzz_constructs_the_distribution_matched_batch(self):
        fan = Fandango(self.AGE_SPEC)
        batch = fan.fuzz(desired_solutions=30)
        self.assertEqual(len(batch), 30)
        ages = [int(str(t).strip()) for t in batch]
        self.assertLessEqual(REDUCERS["normal_fit"](ages, 30, 5), 0.5)

    JOINT_SPEC = (
        '<start> ::= <income> "," <occ> "," <age> "\\n"\n'
        '<income> ::= "0" | "1"\n'
        '<occ> ::= "eng" | "doc" | "art" | "law" | "edu"\n'
        "<age> ::= r'[0-9][0-9]'\n"
        "where fraction(int(<income>) == 1 for x in population) == 0.30\n"
        "where distinct_count(<occ> for x in population) >= 3\n"
        "where normal_fit([int(<age>) for x in population], 30, 5) <= 0.6\n"
    )

    def test_fuzz_constructs_joint_batch(self):
        fan = Fandango(self.JOINT_SPEC)
        batch = fan.fuzz(desired_solutions=30)
        self.assertEqual(len(batch), 30)
        rows = [str(t).strip().split(",") for t in batch]
        self.assertEqual(REDUCERS["fraction"]([int(r[0]) == 1 for r in rows]), 0.30)
        self.assertGreaterEqual(REDUCERS["distinct_count"]([r[1] for r in rows]), 3)
        self.assertLessEqual(REDUCERS["normal_fit"]([int(r[2]) for r in rows], 30, 5), 0.6)

    CORR_SPEC = (
        '<start> ::= <age> "," <income> "\\n"\n'
        "<age> ::= r'[0-9][0-9]'\n"
        "<income> ::= r'[0-9][0-9][0-9]'\n"
        "where correlation((int(<age>), int(<income>)) for x in population) >= 0.5\n"
    )

    def test_fuzz_constructs_correlated_batch(self):
        fan = Fandango(self.CORR_SPEC)
        batch = fan.fuzz(desired_solutions=30)
        self.assertEqual(len(batch), 30)
        rows = [str(t).strip().split(",") for t in batch]
        pairs = [(int(r[0]), int(r[1])) for r in rows]
        self.assertGreaterEqual(REDUCERS["correlation"](pairs), 0.5)

    def test_per_tree_constraint_is_co_enforced_via_fuzz(self):
        # income quota at the batch level + a per-tree lower bound on an independent age field.
        spec = (
            '<start> ::= <income> "," <age> "\\n"\n'
            '<income> ::= "0" | "1"\n'
            "<age> ::= r'[0-9][0-9]'\n"
            "where int(<age>) >= 40\n"
            "where fraction(int(<income>) == 1 for x in population) == 0.30\n"
        )
        fan = Fandango(spec)
        batch = fan.fuzz(desired_solutions=20)
        self.assertEqual(len(batch), 20)
        ages = [int(str(t).strip().split(",")[1]) for t in batch]
        self.assertTrue(all(a >= 40 for a in ages))


if __name__ == "__main__":
    unittest.main()


class TestRegisterRequirement(unittest.TestCase):
    """P3: a custom paired handler registered via register_requirement can be *constructed*
    toward by the sampler (sample) and verified (check), not just used as a soft reducer."""

    def setUp(self):
        random.seed(0)
        # Register a custom "banded uniform" requirement: values drawn evenly across [lo, hi],
        # checked by the Wasserstein distance to a continuous Uniform(lo, hi). This mirrors the
        # built-in uniform_fit but proves the whole extension path (register -> parse -> sample).
        from fandango.constraints.population import register_requirement, wasserstein_fit

        register_requirement(
            "banded_uniform",
            check=lambda values, lo, hi: wasserstein_fit(
                values, lambda p: lo + p * (hi - lo)
            ),
            sample=lambda n, lo, hi: [lo + (i + 0.5) / n * (hi - lo) for i in range(n)],
            allowed_operators=frozenset({"<=", "<"}),
            target_arity=2,
        )

    def tearDown(self):
        # Keep the process-wide registries clean for other tests.
        from fandango.constraints.population import (
            REDUCERS,
            REDUCER_MARGINALS,
            REDUCER_TARGET_ARITY,
            REQUIREMENT_HANDLERS,
        )

        for registry in (
            REDUCERS,
            REDUCER_TARGET_ARITY,
            REDUCER_MARGINALS,
            REQUIREMENT_HANDLERS,
        ):
            registry.pop("banded_uniform", None)

    def test_custom_requirement_is_constructed_and_verified(self):
        grammar = _age_grammar_with(
            "where banded_uniform([int(<age>) for x in population], 10, 40) <= 0.5"
        )
        batch = PopulationSampler(grammar).sample(40)
        self.assertEqual(len(batch), 40)
        from fandango.constraints.population import REDUCERS

        # The custom check gates the batch, exactly like a built-in fit.
        self.assertLessEqual(REDUCERS["banded_uniform"](_ages(batch), 10, 40), 0.5)

    def test_custom_requirement_disallowed_operator_rejected(self):
        grammar = _age_grammar_with(
            "where banded_uniform([int(<age>) for x in population], 10, 40) >= 0.5"
        )
        with self.assertRaises(NotImplementedError):
            PopulationSampler(grammar).sample(40)

    def test_verify_only_handler_is_not_constructible(self):
        # A handler without `sample` is verify-only (soft use); the sampler cannot construct it.
        from fandango.constraints.population import register_requirement

        register_requirement(
            "verify_only_fit",
            check=lambda values, target: abs(sum(values) / max(1, len(values)) - target),
            target_arity=1,
        )
        try:
            grammar = _age_grammar_with(
                "where verify_only_fit([int(<age>) for x in population], 30) <= 1.0"
            )
            with self.assertRaises(NotImplementedError):
                PopulationSampler(grammar).sample(20)
        finally:
            from fandango.constraints.population import (
                REDUCERS,
                REDUCER_TARGET_ARITY,
                REQUIREMENT_HANDLERS,
            )

            for registry in (REDUCERS, REDUCER_TARGET_ARITY, REQUIREMENT_HANDLERS):
                registry.pop("verify_only_fit", None)


class TestPopulationSamplerShortfallPolicy(unittest.TestCase):
    """P4: the on_shortfall knob. fail_loud (default) raises; best_effort returns the closest
    assembled batch with a structured warning instead of raising."""

    def setUp(self):
        random.seed(0)

    def _coarse_normal_grammar(self):
        # Only five possible ages cannot approximate Normal(30, 5) to within 0.5 -> shortfall.
        return _age_grammar_with(
            "where normal_fit([int(<age>) for x in population], 30, 5) <= 0.5",
            age_body='"18" | "25" | "30" | "35" | "45"',
        )

    def test_fail_loud_is_default(self):
        with self.assertRaises(PopulationShortfallError):
            PopulationSampler(self._coarse_normal_grammar()).sample(30)

    def test_best_effort_returns_closest_batch_with_warning(self):
        grammar = self._coarse_normal_grammar()
        sampler = PopulationSampler(grammar, on_shortfall="best_effort")
        with self.assertLogs("fandango", level="WARNING") as cm:
            batch = sampler.sample(30)
        # Returns a full batch (the closest the coarse grammar could assemble), not an exception.
        self.assertEqual(len(batch), 30)
        self.assertTrue(any("best_effort" in m for m in cm.output))

    def test_invalid_policy_rejected(self):
        with self.assertRaises(FandangoValueError):
            PopulationSampler(_grammar_with(), on_shortfall="relax_to_nearest_feasible")
