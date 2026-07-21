#!/usr/bin/env python3
"""Example 9 — honest shortfall handling and the discretization-floor diagnosis.

A hard requirement can be unmeetable, and the sampler never silently returns a wrong batch.
It distinguishes *why* a distributional fit fell short, and lets you choose the policy:

  (a) delta below the DISCRETIZATION FLOOR -> "unsatisfiable in principle". A continuous
      target realized on a gridded field sits at least ~h/4 from it (h = the value step), so
      no placement can get that close. Reported precisely, not as a generic shortfall.

  (b) delta above the floor but the grammar is simply TOO COARSE to reach it -> a shortfall
      that names the achieved fit. With on_shortfall='fail_loud' (default) it raises.

  (c) the same too-coarse case with on_shortfall='best_effort' -> warn and return the closest
      assembled batch instead of raising.

Run:
    PYTHONPATH=src python examples/population_requirements/09_on_shortfall.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fandango import Fandango  # noqa: E402
from fandango.constraints.population_sampler import PopulationShortfallError  # noqa: E402

# Grammar A: five WIDELY spaced ages (step ~5-7) -> floor ~1.25. A small delta is below it.
WIDE = '<start> ::= <age> "\\n"\n<age> ::= "18" | "25" | "30" | "35" | "45"\n'
# Grammar B: five NARROW ages (step 1) -> floor ~0.25, but they cannot cover N(30,5)'s tails,
# so even a delta well above the floor is unreachable -> a genuine "too coarse" shortfall.
NARROW = '<start> ::= <age> "\\n"\n<age> ::= "28" | "29" | "30" | "31" | "32"\n'


def fandango_for(grammar, delta, **kw):
    spec = grammar + f"where normal_fit([int(<age>) for x in population], 30, 5) <= {delta}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".fan", delete=False) as handle:
        handle.write(spec)
        path = handle.name
    return Fandango(open(path))


def main():
    # (a) below the floor: unsatisfiable in principle.
    print("(a) WIDE grammar, delta = 0.5 (below the ~1.25 floor):")
    try:
        fandango_for(WIDE, 0.5).fuzz(desired_solutions=30)
    except PopulationShortfallError as e:
        principle = "unsatisfiable in principle" in str(e)
        print(f"    -> raised; diagnosed as unsatisfiable in principle: {principle}")
        print(f"       {str(e)[:140]}...")

    # (b) above the floor but too coarse, fail_loud (default): raises with the achieved fit.
    print("\n(b) NARROW grammar, delta = 0.5 (above the ~0.25 floor, but unreachable), fail_loud:")
    try:
        fandango_for(NARROW, 0.5).fuzz(desired_solutions=30)
    except PopulationShortfallError as e:
        coarse = "too coarse" in str(e)
        print(f"    -> raised; diagnosed as grammar too coarse: {coarse}")
        print(f"       {str(e)[:140]}...")

    # (c) same case, best_effort: return the closest batch instead of raising.
    print("\n(c) NARROW grammar, delta = 0.5, best_effort:")
    batch = fandango_for(NARROW, 0.5).fuzz(desired_solutions=30, on_shortfall="best_effort")
    print(f"    -> returned {len(batch)} records (the closest achievable) + a warning (above)")


if __name__ == "__main__":
    main()
