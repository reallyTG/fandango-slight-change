#!/usr/bin/env python3
"""Transparent sanity checks for population-level soft objectives.

For each scenario this runs the GA twice on the *same* grammar and seed — once
WITH the population objective and once WITHOUT it (constraints dropped) — then
prints the resulting distribution of the final working set so you can eyeball the
effect. Nothing is hidden: the raw stats, a text histogram, and the objective's
own value are printed for both runs.

Why it inspects the final *population* and not `fandango fuzz -n` output:
  A population objective steers the GA's working set over generations. `fandango
  fuzz` emits solutions from the *stream*, which for soft objectives is dominated
  by early, barely-steered individuals — so `fuzz -n` under-shows the effect (try
  it; see README). The evolved working set (`algorithm.population`) is where the
  steering actually lands, so that is what we measure here.

Run it (no PYTHONPATH needed — the script adds ./src itself):

    python demo/population_objectives/run_sanity.py                 # all scenarios
    python demo/population_objectives/run_sanity.py normal          # one scenario
    python demo/population_objectives/run_sanity.py --generations 600 --seed 3
    python demo/population_objectives/run_sanity.py --attribution uniform   # weaker
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import sys

# --- self-bootstrap: import *this* repo's fandango, not any installed one ----
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from fandango.constraints.population import _normal_fit  # noqa: E402
from fandango.evolution.algorithm import DefaultAlgorithm, LoggerLevel  # noqa: E402
from fandango.language.parse.parse import parse  # noqa: E402

HERE = os.path.dirname(__file__)


def _ages(tree) -> list[int]:
    """Every 2-digit age in one generated table."""
    return [int(m) for m in re.findall(r"(?<!\d)\d\d(?!\d)", str(tree).replace(",", " "))]


def _pairs(tree) -> list[tuple[int, int]]:
    """Every (age, income) row in one generated table."""
    return [(int(a), int(b)) for a, b in re.findall(r"(\d\d),(\d\d)", str(tree))]


def _histogram(values: list[float], width: int = 10, hi: int = 100) -> str:
    bins = [0] * (hi // width)
    for v in values:
        bins[min(int(v) // width, len(bins) - 1)] += 1
    peak = max(bins) or 1
    lines = []
    for i, count in enumerate(bins):
        bar = "#" * round(40 * count / peak)
        lines.append(f"    {i * width:3d}-{i * width + width - 1:<3d} | {bar} {count}")
    return "\n".join(lines)


def _run(spec_path: str, with_objective: bool, *, seed, gens, pop, attribution):
    with open(spec_path) as f:
        grammar, constraints = parse(f, use_stdlib=False, use_cache=False)
    algo = DefaultAlgorithm(
        grammar=grammar,
        constraints=constraints if with_objective else [],
        random_seed=seed,
        logger_level=LoggerLevel.ERROR,
        population_size=pop,
        population_attribution=attribution,
    )
    # Drain the generator so the full generation budget runs, then read the working set.
    list(algo.generate(max_generations=gens))
    return algo.population


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
def scenario_marginal(name, spec, target_desc, seed, gens, pop, attribution, stat):
    print_header(name, spec, target_desc, seed, gens, pop, attribution)
    rows = []
    for label, on in (("WITHOUT objective (baseline)", False), ("WITH objective", True)):
        pop_trees = _run(os.path.join(HERE, spec), on, seed=seed, gens=gens, pop=pop,
                         attribution=attribution)
        ages = [a for t in pop_trees for a in _ages(t)]
        rows.append((label, ages))
        print(f"\n  {label}:")
        print(f"    n={len(ages)}  mean={statistics.mean(ages):.1f}  "
              f"stddev={statistics.pstdev(ages):.1f}  "
              f"normal_fit(30,5)_distance={_normal_fit(ages, 30, 5):.1f}")
        print(_histogram(ages))
    _verdict(stat, rows)


def scenario_joint(name, spec, target_desc, seed, gens, pop, attribution):
    print_header(name, spec, target_desc, seed, gens, pop, attribution)
    results = []
    for label, on in (("WITHOUT objective (baseline)", False), ("WITH objective", True)):
        pop_trees = _run(os.path.join(HERE, spec), on, seed=seed, gens=gens, pop=pop,
                         attribution=attribution)
        pairs = [p for t in pop_trees for p in _pairs(t)]
        try:
            r = statistics.correlation([p[0] for p in pairs], [p[1] for p in pairs])
        except statistics.StatisticsError:
            r = float("nan")
        results.append((label, r))
        sample = "  ".join(f"({a},{b})" for a, b in pairs[:8])
        print(f"\n  {label}:")
        print(f"    n={len(pairs)} rows   corr(age, income) = {r:+.3f}")
        print(f"    sample rows: {sample}")
    base, obj = results[0][1], results[1][1]
    print(f"\n  VERDICT: corr {base:+.3f} -> {obj:+.3f}  (maximizing)  "
          f"{'[UP as expected]' if obj > base else '[no clear lift this seed]'}")
    print("=" * 72)


def print_header(name, spec, target_desc, seed, gens, pop, attribution):
    print("=" * 72)
    print(f"SCENARIO: {name}  —  {target_desc}")
    print(f"spec     : demo/population_objectives/{spec}")
    print(f"settings : population_size={pop}, max_generations={gens}, seed={seed}, "
          f"attribution={attribution}")
    print("measuring: the final GA working set (population) — where soft steering lands")


def _verdict(stat, rows):
    (_, base), (_, obj) = rows
    if stat == "mean":
        b, o = statistics.mean(base), statistics.mean(obj)
        print(f"\n  VERDICT: mean {b:.1f} -> {o:.1f} (target 30)  "
              f"{'[pulled toward target]' if abs(o - 30) < abs(b - 30) else '[no clear pull]'}")
    elif stat == "normal":
        b = _normal_fit(base, 30, 5)
        o = _normal_fit(obj, 30, 5)
        print(f"\n  VERDICT: fit distance to N(30,5) {b:.1f} -> {o:.1f}  "
              f"{'[closer to target shape]' if o < b else '[no clear improvement]'}")
    print("=" * 72)


SCENARIOS = {
    "mean": lambda a: scenario_marginal(
        "mean", "mean_age.fan", "steer mean <age> toward 30",
        a.seed, a.generations, a.population_size, a.attribution, stat="mean"),
    "normal": lambda a: scenario_marginal(
        "normal", "normal_age.fan", "steer <age> toward Normal(30, 5)",
        a.seed, a.generations, a.population_size, a.attribution, stat="normal"),
    "correlation": lambda a: scenario_joint(
        "correlation", "corr_age_income.fan",
        "steer corr(<age>, <income>) upward (joint objective)",
        a.seed, a.generations, a.population_size, a.attribution),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", default="all",
                    help="all (default), or one of: mean, normal, correlation")
    ap.add_argument("--generations", type=int, default=400)
    ap.add_argument("--population-size", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--attribution", choices=["loo", "uniform"], default="loo",
                    help="loo (default) gives a real gradient; uniform barely steers")
    args = ap.parse_args()

    print("These are SOFT objectives: they bias the distribution over generations, they")
    print("do not guarantee it. Expect a clear directional shift, not exact convergence.\n")

    names = list(SCENARIOS) if args.scenario == "all" else [
        n for n in SCENARIOS if n.startswith(args.scenario)]
    if not names:
        ap.error(f"unknown scenario {args.scenario!r}; choose from all, "
                 f"{', '.join(SCENARIOS)}")
    for n in names:
        SCENARIOS[n](args)


if __name__ == "__main__":
    main()
