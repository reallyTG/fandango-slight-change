#!/usr/bin/env python3
"""Example 8 — the grouping policy for multi-valued fields (per_entry vs pool).

When a field can occur several times per record, how should a reducer see those values?
Grouping is a declared property of the *reducer* (set at registration):

    pool       (default) flatten every record's values into one pool — each extra value is
               just another sample. Every built-in reducer uses this.
    per_entry  keep each record's values as a list; the reducer receives list[list] and can
               reduce within-then-across. For a reducer whose input shape is a list-per-record.

This example registers a per_entry reducer that scores how balanced each record's own list
is (a within-record property that a flat pool would erase), and shows the reducer receiving
one list per record.

Run:
    PYTHONPATH=src python examples/population_requirements/08_grouping.py
"""

import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fandango.constraints.population import (  # noqa: E402
    PopulationValue,
    register_reducer,
    try_parse_population_aggregate,
)
from fandango.language.parse.parse import parse  # noqa: E402

CAPTURED = []


def mean_within_record_spread(entries):
    """entries is list[list]: one list of values per record (because grouping='per_entry').
    Return the mean, across records, of each record's own standard deviation — a *within*-
    record statistic that pooling all values into one flat list could never express."""
    CAPTURED.append(entries)  # so we can show the reducer's input shape below
    spreads = [statistics.pstdev(e) for e in entries if len(e) > 1]
    return statistics.mean(spreads) if spreads else 0.0


register_reducer(
    "mean_within_record_spread", mean_within_record_spread, grouping="per_entry"
)

# Each record has THREE ages, so a record yields three values of <age>.
SPEC = """
<start> ::= <age> " " <age> " " <age> "\\n"
<age>   ::= <d> <d>
<d>     ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

minimizing mean_within_record_spread(int(<age>) for x in population)
"""


def main():
    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as handle:
        handle.write(SPEC)
        path = handle.name
    with open(path) as f:
        grammar, constraints = parse(f, use_stdlib=False, use_cache=False)

    soft = constraints[0]
    pv = PopulationValue(
        soft.optimization_goal,
        soft.expression,
        aggregate=try_parse_population_aggregate(soft.expression, soft.searches),
        local_variables=soft.local_variables,
        global_variables=soft.global_variables,
    )

    import random

    random.seed(0)
    pv.evaluate_population([grammar.fuzz() for _ in range(5)])

    first = CAPTURED[0]
    print("grouping='per_entry': the reducer receives ONE LIST PER RECORD (list[list]).")
    print(f"  records evaluated: {len(first)}")
    print(f"  shape of each element: {[type(e).__name__ for e in first[:3]]} ...")
    print(f"  example record's own age list: {first[0]}")
    print("Under the default 'pool' those would be flattened into a single list of ints,")
    print("and a within-record statistic like each record's spread could not be computed.")


if __name__ == "__main__":
    main()
