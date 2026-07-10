#!/usr/bin/env pytest
"""Mechanism B steps 5-6: the population sampler that *constructs* a batch satisfying a hard
population `where`. v1 covers the exact-by-construction `fraction` quota (one boolean per
individual); other shapes raise clear errors."""

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

    def test_non_fraction_reducer_raises(self):
        grammar = _grammar_with(
            "where distinct_count(<income> for x in population) >= 2"
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


class TestPopulationSamplerEndToEnd(unittest.TestCase):
    """Through the public `Fandango.fuzz` API: a population `where` routes to the sampler."""

    SPEC = (
        '<start> ::= <income> "\\n"\n'
        '<income> ::= "0" | "1"\n'
        "where fraction(int(<income>) == 1 for x in population) == 0.30\n"
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

    def test_per_tree_hard_constraint_alongside_requirement_raises(self):
        spec = self.SPEC + "where int(<income>) >= 0\n"
        fan = Fandango(spec)
        with self.assertRaises(NotImplementedError):
            fan.fuzz(desired_solutions=10)


if __name__ == "__main__":
    unittest.main()
