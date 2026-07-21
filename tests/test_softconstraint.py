#!/usr/bin/env pytest

import itertools
import math
import unittest
from collections.abc import Generator

from fandango.errors import FandangoValueError
from fandango.evolution.algorithm import DefaultAlgorithm, LoggerLevel
from fandango.language.parse.parse import parse

from .utils import RESOURCES_ROOT, run_command


class TestSoft(unittest.TestCase):
    @staticmethod
    def get_solutions(
        specification_file,
        desired_solutions,
        random_seed,
        max_generations=500,
        population_size=100,
    ) -> Generator[str, None, None]:
        with open(specification_file, "r") as file:
            grammar_int, constraints_int = parse(
                file, use_stdlib=False, use_cache=False
            )
        assert grammar_int is not None
        assert constraints_int is not None
        fandango = DefaultAlgorithm(
            grammar=grammar_int,
            constraints=constraints_int,
            random_seed=random_seed,
            logger_level=LoggerLevel.DEBUG,
            population_size=population_size,
        )
        generator = itertools.islice(
            fandango.generate(max_generations=max_generations), desired_solutions
        )

        for solution in generator:
            yield str(solution)


class TestSoftValue(TestSoft):
    def test_soft_value(self):
        gen = self.get_solutions(
            RESOURCES_ROOT / "softvalue.fan",
            desired_solutions=50,
            population_size=10,
            random_seed=1,
            max_generations=10000,  # make this a non-limiting factor
        )
        solutions = []
        for s in gen:
            solutions.append(s)
            if "999999-999999" == s:
                return  # optimized solution found, stop generating and don't fail

        self.fail(
            f"999999-999999 not found in the first {len(solutions)} solutions, found: {solutions}"
        )

    def test_min_in_different_contexts(self):
        gen = self.get_solutions(
            RESOURCES_ROOT / "persons_with_constr.fan",
            desired_solutions=60,
            random_seed=1,
        )
        for solution in gen:
            name, age = solution.split(",")
            first_name, last_name = name.split(" ")
            if len(first_name) == 2 and len(last_name) == 2:
                return
        self.fail(
            "No solution found, last first_name: {first_name}, last last_name: {last_name}"
        )

    def test_cli_max_1(self):
        command = [
            "fandango",
            "fuzz",
            "-f",
            str(RESOURCES_ROOT / "persons.fan"),
            "-c",
            "maximizing int(<age>)",
            "-n",
            "20",
            "--max-generations",
            "50",
            "--population-size",
            "10",
            "--random-seed",
            "10",
        ]
        out, err, code = run_command(command)
        lines = [line for line in out.split("\n") if line.strip()]
        self.assertGreater(len(lines), 0, f"\nerr: {err}\nout: {out}")
        last_age = int(lines[-1].split(",")[1])  # e.g., 9999999999999599999999
        self.assertGreater(last_age, 99999999999, f"\nerr: {err}\nout: {out}")
        self.assertEqual(code, 0, code)

    def test_cli_max_2(self):
        command = [
            "fandango",
            "fuzz",
            "-f",
            str(RESOURCES_ROOT / "persons.fan"),
            "--maximize",
            "int(<age>)",
            "-n",
            "20",
            "--max-generations",
            "50",
            "--population-size",
            "10",
            "--random-seed",
            "1",
        ]
        out, err, code = run_command(command)
        lines = [line for line in out.split("\n") if line.strip()]
        last_age = int(lines[-1].split(",")[1])
        self.assertGreater(last_age, 99999999999)
        self.assertEqual(code, 0, code)

    def test_cli_min_1(self):
        command = [
            "fandango",
            "fuzz",
            "-f",
            str(RESOURCES_ROOT / "persons.fan"),
            "-c",
            "minimizing int(<age>)",
            "-n",
            "20",
            "--max-generations",
            "50",
            "--population-size",
            "10",
            "--random-seed",
            "1",
        ]
        out, err, code = run_command(command)
        lines = [line for line in out.split("\n") if line.strip()]
        last_age = int(lines[-1].split(",")[1])
        self.assertEqual(last_age, 0, last_age)
        self.assertEqual(code, 0, code)

    def test_cli_min_2(self):
        command = [
            "fandango",
            "fuzz",
            "-f",
            str(RESOURCES_ROOT / "persons.fan"),
            "--minimize",
            "int(<age>)",
            "-n",
            "20",
            "--max-generations",
            "50",
            "--population-size",
            "10",
            "--random-seed",
            "1",
        ]
        out, err, code = run_command(command)
        lines = [line for line in out.split("\n") if line.strip()]
        last_age = int(lines[-1].split(",")[1])
        self.assertEqual(last_age, 0, last_age)
        self.assertEqual(code, 0, code)


class TestSoftValueEvaluationFailure(TestSoft):
    """Regression tests for the downstream #1 finding: a failed objective evaluation
    must never be scored as optimal (it used to append 0, which wins under `minimizing`)."""

    SPEC = (
        "<start> ::= <x>\n"
        '<x> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"\n\n'
        # honest optimum is x=9 (value 91); raises ZeroDivisionError only at x=0
        "minimizing (100 - int(<x>)) + 0 * (1 // int(<x>))\n"
    )

    def _soft(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".fan", delete=False
        ) as handle:
            handle.write(self.SPEC)
            path = handle.name
        with open(path) as file:
            grammar, constraints = parse(file, use_stdlib=False, use_cache=False)
        return grammar, constraints[0]

    def test_valid_trees_scored_honestly(self):
        grammar, soft = self._soft()
        for digit in "123456789":
            value = soft.fitness(grammar.parse(digit)).fitness()
            self.assertEqual(value, 100 - int(digit))

    def test_failure_substitute_is_goal_aware(self):
        # Unit: a failed evaluation substitutes the worst value for the optimization
        # direction, so a crash is never scored *better* than a valid tree.
        from fandango.constraints.soft import SoftValue

        self.assertEqual(
            SoftValue("min", "1 / 0")._on_evaluation_failure(ZeroDivisionError()),
            math.inf,
        )
        self.assertEqual(
            SoftValue("max", "1 / 0")._on_evaluation_failure(ZeroDivisionError()),
            -math.inf,
        )

    def test_fully_failing_evaluation_raises_not_zero(self):
        # x=0 raises for its only combination -> the objective is broken for that input.
        # It must raise loudly, never return 0.0 (which used to beat every valid tree).
        # Works in both modes: strict (conftest sets FANDANGO_RAISE_ALL_EXCEPTIONS ->
        # the original error propagates) and default (the all-failed guard raises).
        grammar, soft = self._soft()
        with self.assertRaises(Exception):
            soft.fitness(grammar.parse("0")).fitness()

    def test_all_failed_raises_fandango_error_in_default_mode(self):
        # With the strict env var cleared, the shared fitness loop substitutes the
        # sentinel and then raises FandangoValueError because every evaluation failed.
        import os

        grammar, soft = self._soft()
        previous = os.environ.pop("FANDANGO_RAISE_ALL_EXCEPTIONS", None)
        try:
            with self.assertRaises(FandangoValueError):
                soft.fitness(grammar.parse("0")).fitness()
        finally:
            if previous is not None:
                os.environ["FANDANGO_RAISE_ALL_EXCEPTIONS"] = previous
