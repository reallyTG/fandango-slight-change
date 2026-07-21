#!/usr/bin/env python3
"""Example 7 — a CUSTOM hard requirement the sampler can construct toward.

``register_requirement`` is the paired-handler extension point for Mechanism B. Where
``register_reducer`` adds a reducer the sampler can only *verify*, ``register_requirement``
also supplies a ``sample`` constructor, so the sampler can *build* a batch that satisfies
the requirement by construction:

    check(values, *params)  -> the batch-level aggregate (also usable as a soft reducer)
    sample(n, *params)      -> n target values the sampler pins a column to
    floor(values, *params)  -> optional: the smallest achievable check value (a diagnosis)

Registration is process-wide and must run BEFORE the spec is parsed.

This example registers ``triangular_fit`` — matching a column to a symmetric triangular
distribution on [lo, hi] — a distribution Fandango does not ship. Run:

    PYTHONPATH=src python examples/population_requirements/07_register_requirement.py
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fandango import Fandango  # noqa: E402
from fandango.constraints.population import (  # noqa: E402
    register_requirement,
    wasserstein_fit,
)


def triangular_quantile(lo, hi):
    """Inverse CDF of a symmetric triangular distribution on [lo, hi] (mode at the midpoint)."""
    mid = (lo + hi) / 2.0

    def q(p):
        if p < 0.5:
            return lo + math.sqrt(p * (hi - lo) * (mid - lo))
        return hi - math.sqrt((1 - p) * (hi - lo) * (hi - mid))

    return q


# check: distance of the column to the target (a fit is a distance -> only <=/< make sense).
# sample: draw n values on the target's quantiles so the sampler can pin a column to them.
register_requirement(
    "triangular_fit",
    check=lambda values, lo, hi: wasserstein_fit(values, triangular_quantile(lo, hi)),
    sample=lambda n, lo, hi: [
        triangular_quantile(lo, hi)((i + 0.5) / n) for i in range(n)
    ],
    allowed_operators=frozenset({"<=", "<"}),
    target_arity=2,
)

SPEC = """
<start> ::= <v> "\\n"
<v>     ::= <d> <d>
<d>     ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

where triangular_fit([int(<v>) for x in population], 10, 60) <= 1.0
"""


def main():
    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as handle:
        handle.write(SPEC)
        path = handle.name

    with open(path) as f:
        batch = Fandango(f).fuzz(desired_solutions=200)

    values = sorted(int(str(t).strip()) for t in batch)
    quantile = triangular_quantile(10, 60)
    distance = wasserstein_fit(values, quantile)
    print(f"custom triangular_fit requirement, N={len(values)}")
    print(f"  achieved fit distance: {distance:.3f}  (required <= 1.0)")
    print(f"  value range: {values[0]}..{values[-1]} (target support [10, 60], mode ~35)")
    # A triangular distribution peaks at the mode: values cluster near the middle.
    middle = sum(1 for v in values if 27 <= v <= 43)
    print(f"  {middle}/{len(values)} values within +/-8 of the mode (35) — the peak")


if __name__ == "__main__":
    main()
