# Population-objective sanity checks

Transparent, eyeball-able checks that population-level soft objectives actually
steer the distribution of generated data. Each scenario runs the GA twice on the
**same grammar and seed** — once WITH the objective, once WITHOUT — and prints the
resulting distribution of the final population, so you can see the shift directly.

## Run it

No `PYTHONPATH` needed — the runner puts this repo's `src` on the path itself:

```shell
python demo/population_objectives/run_sanity.py               # all three scenarios
python demo/population_objectives/run_sanity.py mean          # just one
python demo/population_objectives/run_sanity.py correlation
python demo/population_objectives/run_sanity.py --generations 600 --seed 3
python demo/population_objectives/run_sanity.py --attribution uniform   # weaker, for comparison
```

## Files

| File | What it steers |
| --- | --- |
| [`mean_age.fan`](mean_age.fan) | mean `<age>` → 30 (single moment) |
| [`normal_age.fan`](normal_age.fan) | `<age>` shape → `Normal(30, 5)` (distributional fit) |
| [`corr_age_income.fan`](corr_age_income.fan) | `corr(<age>, <income>)` upward (joint, per-row) |
| [`run_sanity.py`](run_sanity.py) | the runner (with/without, stats + histograms) |

## What to expect

These are **soft** objectives: they *bias* the distribution over generations, they
do not guarantee it. Look for a clear directional shift, not exact convergence.
Representative run (`--generations 300`, seed 1):

- **mean**: mean age `48.4 → 42.7` (baseline → objective; target 30)
- **correlation**: `corr −0.133 → +0.265`; the WITH-objective sample rows co-vary,
  e.g. `(99,97) … (0,1)`
- **normal**: fit-distance to N(30,5) drops slowly — the weakest of the three (see below)

Strength depends on two knobs:
- **`--generations`**: more generations → stronger bias. `normal` in particular needs
  a large budget (matching both centre *and* spread is harder than one moment).
- **`--attribution`**: `loo` (default) gives a real per-individual gradient; `uniform`
  barely steers — run it to see the difference.

## Why this measures the *population*, not `fandango fuzz -n`

A population objective steers the GA's **working set**. `fandango fuzz` emits solutions
from the *stream*, which for soft objectives is dominated by early, barely-steered
individuals — so `fuzz -n` badly under-shows the effect. You can confirm this yourself:

```shell
# emitted stream — barely shifts (≈1 point):
PYTHONPATH=src python -c "import sys;from fandango.cli import main;sys.argv=['fandango','fuzz','-f','demo/population_objectives/mean_age.fan','-n','60','--max-generations','400','--population-size','40','--random-seed','1'];main()" \
  | awk 'NF{s+=$1;n++} END{printf "emitted mean=%.1f over n=%d\n",s/n,n}'
```

That prints ~48 with the objective vs ~49 without — almost no visible steering, while
the working set (what `run_sanity.py` measures) moves several points. This is the
documented working-set-vs-output-stream gap; see the "Scope of `population`" admonition
in [`docs/Distributions.md`](../../docs/Distributions.md).

## Where the code lives

- Reducers, fits, `register_reducer`, and row-scoped joint objectives:
  [`src/fandango/constraints/population.py`](../../src/fandango/constraints/population.py)
- User docs: [`docs/Distributions.md`](../../docs/Distributions.md)
- Unit/e2e tests: [`tests/test_populationconstraint.py`](../../tests/test_populationconstraint.py)
