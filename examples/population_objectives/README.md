# Population-level soft objectives — examples

These `.fan` files demonstrate **Mechanism A: soft population-level objectives** —
`minimizing`/`maximizing` an aggregate computed over the *whole generated
population*, not a single record. The GA is steered (best-effort) toward the goal.

## Syntax in one line

```
minimizing abs(mean(int(<age>) for x in population) - 40)
```

- `population` is a **reserved binder** ranging over every generated record.
- The generator (`... for x in population`) produces one value per record;
  a **reducer** collapses them into the aggregate that is optimised.
- Supported reducers: `mean`, `stddev`, `fraction`, `count`, `distinct_count`,
  `correlation`, and distribution fits like `normal_fit` / `exponential_fit`.

## Running an example

```
PYTHONPATH=src fandango fuzz -f examples/population_objectives/01_mean_age.fan -n 20
```

Useful knobs: `-N` (max generations), `--population-size`,
`--population-attribution {loo,uniform,marginal}`.
Drop the `minimizing`/`maximizing` line from any file to see the unsteered baseline.

## The examples

| File | Objective | Steers well? |
|------|-----------|--------------|
| `01_mean_age.fan` | Pull a column's mean to a target | ✅ location |
| `02_category_fraction.fan` | Bias a category's share (~50% F) | ✅ proportion |
| `03_distinct_count.fan` | Maximise variety across records | ✅ between-record |
| `04_normal_fit.fan` | Fit a column's shape to N(30,5) | ⚠️ centre yes, spread no |
| `05_correlation_joint.fan` | Correlate two columns per row | ✅ between-record |

## Important caveat — what soft objectives can and can't do

These are a **soft bias, not a guarantee**. In practice they steer **location**
(where a distribution sits — mean, proportion, correlation, variety) reliably, but
they do **not** reliably steer **within-record spread** (e.g. matching a target
variance). Selection acts on whole records, so dispersion that lives *inside* each
record isn't attributable to any individual and can't be selected for. See
`04_normal_fit.fan`'s header and `docs/Distributions.md` for the full explanation.

If you need a distribution matched *as a hard requirement* (shape included), that's
a different mechanism (sample-then-structure) — see the population-steering plans in
the repo root.
