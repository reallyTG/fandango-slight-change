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

    def test_multiple_requirements_raise(self):
        grammar = _grammar_with(
            "where fraction(int(<income>) == 1 for x in population) == 0.30",
            "where fraction(int(<income>) == 0 for x in population) >= 0.10",
        )
        with self.assertRaises(NotImplementedError):
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

    def test_per_tree_hard_constraint_alongside_requirement_raises(self):
        spec = self.SPEC + "where int(<income>) >= 0\n"
        fan = Fandango(spec)
        with self.assertRaises(NotImplementedError):
            fan.fuzz(desired_solutions=10)


if __name__ == "__main__":
    unittest.main()
