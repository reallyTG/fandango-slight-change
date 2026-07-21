#!/usr/bin/env python3
"""Example 6 — observe a SOFT population objective via fuzz(return_population=True).

A soft population objective (Mechanism A) *steers* the GA's working set, but plain
``fuzz()`` returns the solution *stream*, which is dominated by early, barely-steered
individuals — so a working objective can look like no objective at all. Pass
``return_population=True`` to run the GA to completion and get the steered working set
(the ``population`` property exposes the same set). This script shows the difference.

Run:
    PYTHONPATH=src python examples/population_requirements/06_return_population.py
"""

import os
import statistics
import sys

# Make this repo's source importable regardless of the ambient install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fandango import Fandango  # noqa: E402

# A 5-row table of ages; the objective pulls the population's mean age toward 30.
SPEC = """
<start>  ::= <row> "\\n" <row> "\\n" <row> "\\n" <row> "\\n" <row>
<row>    ::= <age>
<age>    ::= <digit> <digit>
<digit>  ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

minimizing abs(mean(int(<age>) for x in population) - 30)
"""

TARGET = 30


def ages(trees):
    out = []
    for tree in trees:
        out.extend(int(line) for line in str(tree).split("\n") if line.strip())
    return out


def mean_age(trees):
    return statistics.mean(ages(trees))


def main():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as handle:
        handle.write(SPEC)
        path = handle.name

    settings = dict(max_generations=300, population_size=40, random_seed=1)

    # (a) The default stream: what fuzz() returns without return_population.
    with open(path) as f:
        stream = Fandango(f).fuzz(desired_solutions=40, **settings)

    # (b) The steered working set: run the GA to completion and return it.
    with open(path) as f:
        population = Fandango(f).fuzz(return_population=True, **settings)

    print(f"target mean age:            {TARGET}")
    print(f"fuzz() stream mean age:     {mean_age(stream):5.1f}   <- barely steered")
    print(f"return_population mean age: {mean_age(population):5.1f}   <- the objective at work")


if __name__ == "__main__":
    main()
