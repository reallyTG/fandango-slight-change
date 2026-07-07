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

`register_reducer(name, reducer, *, target_arity=0)` adds `reducer(values, *target_params)`
to the registry under `name`; `target_arity` is how many trailing literal arguments the
objective supplies (the distribution's parameters). Any reducer that returns a float works
— it need not be a distributional fit — but for one that is, `wasserstein_fit` + a quantile
function is the whole recipe. Closed-form quantiles (logistic, Weibull, Pareto, triangular,
…) are one-liners; distributions without one (gamma, beta) come for free through
`scipy.stats.<dist>.ppf`, which is why the richer catalogue is best kept in the
Scipy-carrying downstream rather than here.

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
  moves the aggregate toward the goal. This gives a real selection gradient.
* **`uniform`** gives every input the same score. It creates no gradient _between_ inputs,
  so it barely steers; it is mainly useful as a baseline for comparison.

Select the mode with the `--population-attribution {loo,uniform}` command-line flag (or the
`population_attribution` constructor argument):

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

This mechanism generalizes the built-in [diversity](sec:diversity) bonus, which is a
fixed, population-aware objective baked into the search; population-level objectives let
you express your own.

```{versionadded} 1.x
Population-level soft objectives are an experimental addition.
```
