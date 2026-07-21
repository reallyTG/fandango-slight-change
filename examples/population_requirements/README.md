# Population-level hard requirements — examples

These examples demonstrate **Mechanism B: hard population-level requirements** — a `where`
clause that *guarantees* a property over the whole emitted batch of N, **constructed** by a
sampler above the genetic algorithm rather than steered toward. This is the hard counterpart
to the soft objectives in [`../population_objectives/`](../population_objectives/) (which
*bias* the distribution, best-effort).

## Syntax in one line

```
where fraction(int(<income>) == 1 for x in population) == 0.30
```

Any `where` that mentions the reserved **`population`** binder is promoted to a hard
requirement. Because it is a guarantee over the *batch*, you must request a fixed size
(`-n N` / `desired_solutions=N`). Per-record `where` constraints (no `population`) remain
hard validity rules and are co-enforced.

## Running the `.fan` examples

```
PYTHONPATH=src fandango fuzz -f examples/population_requirements/01_fraction_quota.fan -n 20
```

The `.py` examples are self-contained (they add `src` to the path themselves):

```
python examples/population_requirements/06_return_population.py
```

Pin `PYTHONHASHSEED=0` for byte-reproducible runs.

## The examples

### Declarative — hard `where` in a spec

| File | Feature | Guarantee |
|------|---------|-----------|
| `01_fraction_quota.fan` | `fraction(...) == p` | exact proportion by construction (snap-and-warn at awkward N) |
| `02_distinct_count.fan` | `distinct_count(...) >= K` | exact number of distinct values |
| `03_distribution_fit.fan` | `normal_fit(...) <= δ` | column shape within a Wasserstein tolerance; **discretization floor** |
| `04_correlation.fan` | `correlation((<x>,<y>)) == r` | two columns coupled at a target correlation (copula search) |
| `05_combined.fan` | several requirements + per-record `where` | disjoint-field requirements co-enforced together |

### Programmatic — the Python API

| File | Feature | Shows |
|------|---------|-------|
| `06_return_population.py` | `fuzz(return_population=True)` | soft objectives made observable (working set vs stream) |
| `07_register_requirement.py` | `register_requirement(...)` | a custom `triangular_fit` the sampler *constructs* toward |
| `08_grouping.py` | `grouping="per_entry"` | a reducer receiving one list per record (`list[list]`) |
| `09_on_shortfall.py` | `on_shortfall` + the floor | floor diagnosis vs too-coarse shortfall vs best-effort |

## What each requirement guarantees

| Requirement | Guarantee | Notes |
|---|---|---|
| `fraction(<pred>) == p` (`>=`/`<=` too) | exact by construction | `== p` snaps to `round(p·N)/N` and warns when `p·N` isn't integral |
| `distinct_count(<field>) OP K` | exact integer count | reach (`>=`/`>`/`==`) or cap (`<=`/`<`); needs the grammar to have K values and N ≥ K |
| `*_fit(...) <= δ` | within δ by construction | δ is a 1-Wasserstein distance; `<=`/`<` only; δ must stay above the discretization floor |
| `correlation((<x>,<y>)) OP r` | toward r | inequalities drive to ±1; `== r` targets r within `correlation_tolerance` (default 0.15) |

Requirements combine when their fields are **disjoint**; same-field / nested-field sets are
rejected with a clear error.

## Honest limits (v1)

- **Per-row / conditional `P(y|x)`** is not constructed field-by-field yet: emit the whole
  correlated tuple from one **umbrella symbol**'s `:=` generator so the coupling is baked in.
  `>2`-way coupling and stratified/grouped quotas are future work.
- **Chained per-record comparisons** (`18 <= int(<age>) <= 65`) are not co-enforced by the
  sampler — use two simple `where` lines (`>= 18` and `<= 65`), as in `05_combined.fan`.
- `per_entry`/`per_row` *construction* (only the soft path honors grouping today) and a
  per-spec `grouping=` call syntax are future work.

See [`docs/Distributions.md`](../../docs/Distributions.md) (“Hard Population-Level
Requirements”) and [`docs/adult-hard.fan`](../../docs/adult-hard.fan) for the full reference.
