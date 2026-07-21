---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec:distributions)=
# Statistical Distributions

When introducing [generators](Generators.md), we have seen a first method on how to create distributions.
Generators shape the value of a _single_ field in a _single_ input.
This chapter is about shaping a property of the _whole set_ of generated inputs — for
example, "make the _average_ `<age>` across all generated people come out around 30", or
"make the generated values as _varied_ as possible".

## Population-Level Soft Objectives

A regular `minimizing`/`maximizing` [soft constraint](sec:soft-constraints) scores each
input on its own.
A _population-level_ objective instead scores an _aggregate_ over the current working set
of inputs, and gently steers the search so that the aggregate approaches your goal.

You write one by aggregating over the reserved binder `population` with an aggregate
helper:

```
<start>  ::= <person>+
<person> ::= <name> "," <age> "\n"
<name>   ::= r'[a-z]+'
<age>    ::= r'[0-9]+'

minimizing abs(mean(int(<age>) for x in population) - 30)
```

Here:

* `population` is a **reserved identifier** that stands for the set of inputs Fandango is
  currently evolving. You may only use it as the thing a helper iterates over, i.e.
  `helper(<inner> for x in population)`.
* `<inner>` (here `int(<age>)`) is an ordinary per-input expression: it uses the same
  non-terminal search machinery as any other constraint, evaluated against each input.
* `mean(...)` is an **aggregate helper** (see below) that reduces the per-input values to
  a single number.
* The surrounding expression (`abs(... - 30)`) is what actually gets optimized once the
  aggregate has been computed — so this objective drives the mean age toward 30.

Maximizing works the same way; for instance, to bias generation toward larger ages:

```
maximizing mean(int(<age>) for x in population)
```

### Aggregate helpers

| Helper                    | Meaning                                             |
| ------------------------- | --------------------------------------------------- |
| `mean(...)`               | Arithmetic mean of the per-input values             |
| `stddev(...)`             | Population standard deviation                        |
| `count(...)`              | Number of values                                    |
| `distinct_count(...)`     | Number of _distinct_ values                         |
| `fraction(...)`           | Fraction of values that are truthy                  |
| `normal_fit(..., mu, sigma)`      | Distance to a normal `N(mu, sigma)`         |
| `lognormal_fit(..., mu, sigma)`   | Distance to a log-normal (skewed, positive) |
| `uniform_fit(..., lo, hi)`        | Distance to a flat `Uniform(lo, hi)`        |
| `exponential_fit(..., rate)`      | Distance to an `Exponential(rate)` (mean `1/rate`) |

`fraction` pairs naturally with a boolean inner expression, e.g.
`maximizing fraction(int(<age>) >= 18 for x in population)` to bias toward adults.

### Fitting a whole distribution

The moment helpers above steer one _statistic_ at a time. The `*_fit` helpers instead
steer the _shape_ of a column toward a target distribution, by measuring the
[1-Wasserstein / earth-mover distance](https://en.wikipedia.org/wiki/Wasserstein_metric)
from the population's values to the target (in the same units as the values — zero when
the values sit exactly on the target's quantiles). Minimizing that distance biases the
column toward the target shape:

```
minimizing normal_fit([int(<age>) for x in population], 30, 5)
```

reads as "make `<age>` across the population look like `N(30, 5)`". Swap the helper to aim
at a different shape — `lognormal_fit(..., 3, 1)`, `uniform_fit(..., 18, 65)`,
`exponential_fit(..., 0.05)`.

These four built-ins are deliberately a small, standard-library-only *sample*. Fandango
does not try to ship a full catalogue of distributions; instead it exposes the machinery
so the tool that embeds Fandango can register exactly the distributions it needs.

### Joint objectives across columns (experimental)

The objectives above steer one column. To steer a *relationship between* columns — the
kind of joint structure that is everywhere in tabular data — an objective may combine more
than one field in its inner expression, and it is then evaluated **per row** so the fields
stay paired:

```
maximizing correlation((int(<age>), int(<income>)) for x in population)
```

`correlation` is a joint reducer over the `(age, income)` pairs, one per row, pooled across
the population; the objective biases the two columns to co-vary. Any objective whose inner
expression references two or more distinct non-terminals is treated as joint this way (for
example `minimizing abs(mean(int(<a>) - int(<b>) for x in population) - 5)` steers a
per-row difference).

```{admonition} How the pairing works — and its limits
:class: attention
A single input is a whole table of many rows, so "pair `<age>` with `<income>`" only makes
sense *within a row*. Fandango infers the row non-terminal automatically — the tightest one
whose every instance holds exactly one of each referenced field (here `<person>`) — and
evaluates the inner expression against each such subtree. This is what stops the fields from
being combined as a meaningless cross product, which would flatten any correlation to zero.

It is a **prototype**, with the same *soft*, best-effort character as the rest of this
chapter, plus two constraints: the row unit must be inferable (each field appears exactly
once per row), and joint reducers such as `correlation` are the extension surface — richer
targets (a joint density via an optimal-transport distance, mutual information, a copula
fit) plug in through `register_reducer` just like the marginal fits.
```

### Adding your own distribution

Each `*_fit` helper is just the shared 1-Wasserstein distance
(`fandango.constraints.population.wasserstein_fit`) plus that distribution's *quantile
function* (inverse CDF). To add one from your own code, call `register_reducer` **before**
the spec that uses it is parsed:

```python
from scipy.stats import gamma
from fandango.constraints.population import register_reducer, wasserstein_fit

register_reducer(
    "gamma_fit",
    lambda values, a, scale: wasserstein_fit(values, lambda p: gamma.ppf(p, a, scale=scale)),
    target_arity=2,
)
```

after which a spec can use it exactly like a built-in:

```
minimizing gamma_fit([int(<amount>) for x in population], 2.0, 10.0)
```

`register_reducer(name, reducer, *, target_arity=0, marginal=None)` adds
`reducer(values, *target_params)` to the registry under `name`; `target_arity` is how many
trailing literal arguments the objective supplies (the distribution's parameters). Any
reducer that returns a float works — it need not be a distributional fit — but for one that
is, `wasserstein_fit` + a quantile function is the whole recipe. Closed-form quantiles
(logistic, Weibull, Pareto, triangular, …) are one-liners; distributions without one (gamma,
beta) come for free through `scipy.stats.<dist>.ppf`, which is why the richer catalogue is
best kept in the Scipy-carrying downstream rather than here.

For a distributional fit, prefer `register_distribution_fit`, which wires **both** the
reducer and its `marginal` companion (see [Attribution](#attribution)) from a single
quantile — so your distribution gets the sharper `marginal` gradient for free:

```python
from scipy.stats import gamma
from fandango.constraints.population import register_distribution_fit

register_distribution_fit(
    "gamma_fit",
    lambda a, scale: lambda p: gamma.ppf(p, a, scale=scale),
    target_arity=2,
)
```

The lower-level `register_reducer` also takes an optional `marginal=` companion directly; omit
it and objectives using that reducer simply fall back to `loo` attribution.

```{admonition} Bracket the generator when a helper takes target parameters
:class: note
`normal_fit` takes its target (`mu`, `sigma`) as trailing arguments after the values.
Python only lets a bare generator expression stand alone as a *single* argument, so once
there are extra arguments you must bracket it — use the list form `[... for x in
population]` (as above) or parenthesize it `(... for x in population)`. Helpers with no
target parameters (`mean`, `stddev`, …) can still take the bare `... for x in population`.
```

Like every population-level objective this is a *soft* guide: it biases selection and
tightens the fit over successive generations rather than snapping to it, and (as with the
attribution note above) how hard it pulls depends on `--population-attribution`. Fitting
both location and spread is a harder target than a single moment, so expect it to need
more generations to settle.

### Attribution

An objective yields one aggregate number per generation, but selection needs a
per-individual contribution. How that one number is spread back onto individuals — the
_attribution_ — is what determines how strongly the objective steers:

* **`loo`** (leave-one-out, the default) rewards each input by how much _including_ it
  moves the aggregate toward the goal, by re-evaluating the aggregate over the population
  minus that input. This gives a real selection gradient, at O(n) re-aggregations.
* **`marginal`** is a cheaper, sharper O(N) approximation of `loo`: instead of
  re-aggregating, it asks the reducer for each value's analytic removal influence and scores
  the objective at the linearly-perturbed aggregate. It is available for every reducer that
  ships a `marginal` companion (all the built-ins except `correlation`, plus anything added
  via `register_distribution_fit`), and falls back to `loo` for the rest. In practice it
  steers _location_ objectives (e.g. `mean`) a little harder than `loo` and is roughly on par
  elsewhere. Note it cannot, on its own, tighten a distribution's _spread_: selection acts on
  whole inputs, and spread is a property of the pooled values rather than of any one input —
  so both `marginal` and `loo` shift where the distribution sits more readily than how wide
  it is.
* **`uniform`** gives every input the same score. It creates no gradient _between_ inputs,
  so it barely steers; it is mainly useful as a baseline for comparison.

Select the mode with the `--population-attribution {loo,marginal,uniform}` command-line flag
(or the `population_attribution` constructor argument):

```shell
$ fandango fuzz -f persons.fan --population-attribution loo
```

```{admonition} Best-effort, not a guarantee
:class: attention
Population-level objectives are **soft**: they *steer* the distribution of generated
inputs, but they do not *guarantee* it. They only shape the set of otherwise-valid inputs,
and their strength depends on the attribution mode (above). Treat them as a bias, not an
invariant.
```

```{admonition} Scope of `population`
:class: note
`population` refers to the genetic algorithm's current working set (of size
`population_size`), not the full stream of inputs Fandango ultimately emits. It is
therefore an approximation of the true output distribution.
```

```{admonition} Reproducibility needs `PYTHONHASHSEED`, not just `random_seed`
:class: warning
The search reads dictionary and set iteration order, which Python salts per process.
Setting `random_seed` alone makes a run deterministic *within* one process but **not**
across runs — the same command with the same `random_seed` can drift (e.g. a baseline
mean of 46.9 on one run and 53.3 on the next). For byte-identical results, also fix
`PYTHONHASHSEED` *before* the interpreter starts:

```shell
$ PYTHONHASHSEED=0 fandango fuzz -f persons.fan --random-seed 1 ...
```

This matters most when comparing distributions across runs (A/B measuring how hard an
objective steers). It applies to both soft objectives and the hard `where` sampler, whose
candidate shuffles likewise depend on the caller having seeded `random` and pinned
`PYTHONHASHSEED`.
```

This mechanism generalizes the built-in [diversity](sec:diversity) bonus, which is a
fixed, population-aware objective baked into the search; population-level objectives let
you express your own.

```{versionadded} 1.x
Population-level soft objectives are an experimental addition.
```

## Hard Population-Level Requirements

A soft objective *steers* the distribution; a **hard population `where`** *guarantees* it
over the emitted batch. Any `where` that mentions the reserved `population` binder is
promoted to a population requirement, routed **around** the genetic algorithm to a sampler
that *constructs* a batch of the requested size — you must pass a fixed `-n N` /
`desired_solutions=N`. Per-record `where` constraints still apply and are co-enforced (every
constructed individual satisfies them). A full runnable example is `docs/adult-hard.fan`.

```
# exactly 24% of records have income == 1 (a quota, exact by construction)
where fraction(int(<income>) == 1 for x in population) == 0.24

# at least 12 distinct occupations appear across the batch
where distinct_count(<occupation> for x in population) >= 12

# the <age> column matches Normal(38, 13) within a Wasserstein tolerance
where normal_fit([int(<age>) for x in population], 38, 13) <= 0.7

# two columns correlate at (approximately) the target
where correlation((int(<education_num>), int(<hours>)) for x in population) >= 0.3
```

### What each shape guarantees

| Requirement | Guarantee | Notes |
|---|---|---|
| `fraction(<pred>) == p` (`>=`/`<=` too) | exact by construction | `== p` snaps to the nearest achievable `round(p·N)/N` and warns when `p·N` is non-integral |
| `distinct_count(<field>) OP K` | exact integer count | reach (`>=`/`>`/`==`) or cap (`<=`/`<`) |
| `normal_fit`/`lognormal_fit`/`uniform_fit`/`exponential_fit(...) <= δ` | within δ by construction | δ is a 1-Wasserstein distance; only `<=`/`<` are meaningful |
| `correlation((<x>,<y>)) OP r` | toward r | `>=`/etc. drive to the ±1 extreme; `== r` targets r via a copula search within `correlation_tolerance` |

Requirements are combined when they target **disjoint fields** (each field is constrained by
at most one requirement); same-field and nested-field sets are rejected with a clear error.

```{admonition} The discretization floor (why δ can't be arbitrarily small)
:class: note
A continuous target realized on an **integer** (or otherwise gridded) field sits a fixed
distance from the true continuous distribution no matter how the values are placed — about
`h/4` for a grid step `h` (≈ 0.25 for integers). A `δ` below that floor is *unsatisfiable in
principle*, and the sampler says so precisely rather than reporting a generic shortfall. Keep
`δ` above the floor.

On this by-construction path the batch's order statistics are *placed* on the target
quantiles, so the gate is the achieved point-estimate distance (a regression check that
structuring didn't perturb the draw), not a bootstrap confidence bound — resampling a
deliberately quantile-placed batch would wildly overstate the distance. The bootstrap
equivalence test in `statistics/equivalence.py` is there for a future subset-selection path,
where the sample *is* an i.i.d. draw.
```

### Shortfall policy

If a requirement can't be met (a grammar too coarse to reach `δ`, an infeasible cell),
`on_shortfall` decides what happens: `fail_loud` (default) raises with a precise diagnosis;
`best_effort` warns and returns the closest assembled batch. From the CLI:

```shell
$ fandango fuzz -f adult-hard.fan -n 200 --on-shortfall best_effort
```

### Custom requirements

`register_requirement` adds a requirement the sampler can *construct* toward — the paired
counterpart to `register_reducer` (which only verifies). Supply a `check` (the batch-level
aggregate, also usable as a soft reducer) and a `sample(n, *params)` that returns `n` target
values the sampler pins a column to; an optional `floor` gives the discretization diagnosis:

```python
from fandango.constraints.population import register_requirement

register_requirement(
    "banded_uniform",
    check=lambda values, lo, hi: uniform_fit_distance(values, lo, hi),
    sample=lambda n, lo, hi: [lo + (i + 0.5) / n * (hi - lo) for i in range(n)],
    allowed_operators=frozenset({"<=", "<"}),
    target_arity=2,
)
# then, in a spec parsed after this call:
#   where banded_uniform([int(<age>) for x in population], 10, 40) <= 0.5
```

A reducer may also declare a `grouping` policy for multi-valued fields: `pool` (default,
flatten every individual's values into one pool), or `per_entry` (the reducer receives one
list per individual — `list[list]`; honored by the soft path). Registration is process-wide
and must run **before** the spec is parsed.

### Honest limits (v1)

- **Per-row coupling / conditional `P(y|x)`** is not yet constructed field-by-field: emit the
  whole correlated tuple from one **umbrella symbol**'s `:=` generator instead (so the coupling
  is baked in by construction). `>2`-way coupling and stratified/grouped quotas are future work.
- **Per-record chained comparisons** (`17 <= int(<age>) <= 90`) are not co-enforced by the
  sampler; use two simple `where` lines (`>= 17` and `<= 90`).
- The multiplicity `per_entry`/`per_row` *construction* path (only the soft path honors them
  today) and per-spec `grouping=` call syntax are future work.

```{versionadded} 1.x
Hard population-level requirements (Mechanism B) are an experimental addition.
```
