"""Mechanism A — soft population-level objectives.

A normal ``SoftValue`` scores a *single* ``DerivationTree`` via ``fitness(tree)``.
A :class:`PopulationValue` instead expresses a *soft aggregate objective* over the
whole population, e.g.::

    minimizing abs(mean(int(<age>) for x in population) - 30)

Here ``population`` is a reserved binder (the current GA working set), the element
expression (``int(<age>)``) reuses all the normal per-tree search machinery, and the
call wrapping the generator (``mean``) is a registered *reducer*. The surrounding
expression (``abs(... - 30)``) is the actual quantity being optimized once the
aggregate has been substituted in.

This module is intentionally self-contained (reducers + parser + value type). It does
*no* evaluator wiring — that is a separate step.
"""

import ast
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from statistics import NormalDist, StatisticsError, correlation
from typing import Any, Optional

from fandango.constraints.comparison import ComparisonConstraint
from fandango.constraints.constraint import Constraint
from fandango.constraints.failing_tree import Comparison
from fandango.constraints.fitness import ValueFitness
from fandango.constraints.soft import SoftValue, Value
from fandango.errors import FandangoValueError
from fandango.language.search import NonTerminalSearch
from fandango.language.tree import DerivationTree
from fandango.logger import LOGGER

# The reserved identifier a population objective iterates over.
POPULATION_BINDER = "population"
# Placeholder the aggregate value is substituted for when evaluating the outer expression.
AGGREGATE_PLACEHOLDER = "___fandango_population_agg___"


# --------------------------------------------------------------------------- #
# Reducers
# --------------------------------------------------------------------------- #
def _mean(values: list[Any]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[Any]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _fraction(values: list[Any]) -> float:
    """Fraction of truthy values — pairs with a boolean inner expression."""
    return sum(1 for v in values if v) / len(values) if values else 0.0


def _distinct_count(values: list[Any]) -> int:
    return len(set(values))


def _count(values: list[Any]) -> int:
    return len(values)


def _correlation(pairs: list[Any]) -> float:
    """Pearson correlation of a list of ``(x, y)`` pairs — a *joint* reducer.

    Its inner expression yields one pair per row, e.g.
    ``correlation((int(<age>), int(<income>)) for x in population)``. That only works
    because the objective is evaluated row-by-row rather than cross-producted over the
    whole tree; see the row-scoping note on :class:`PopulationValue`. Returns 0.0 when
    the correlation is undefined (fewer than two pairs, or a column has no variance)."""
    xs: list[float] = []
    ys: list[float] = []
    for p in pairs:
        try:
            a, b = p
        except (TypeError, ValueError):
            raise FandangoValueError(
                "correlation expects a joint inner expression yielding (x, y) pairs, "
                "e.g. correlation((int(<age>), int(<income>)) for x in population); "
                f"got {p!r}."
            )
        xs.append(float(a))
        ys.append(float(b))
    if len(xs) < 2:
        return 0.0
    try:
        return correlation(xs, ys)
    except StatisticsError:  # a constant column has no correlation
        return 0.0


# --- distributional-fit reducers ------------------------------------------- #
# A distributional fit is the 1-Wasserstein (earth-mover) distance from the empirical
# distribution of the values to a target distribution. The *only* per-distribution piece
# is the target's quantile function (inverse CDF); everything else is shared. The handful
# of fits below are built-in *examples* backed by the stdlib; the full catalogue of
# distributions is meant to live in the downstream importer via `register_reducer` +
# `wasserstein_fit` (e.g. wrapping `scipy.stats.<dist>.ppf`). See `register_reducer`.
def wasserstein_fit(values: list[Any], quantile: Callable[[float], float]) -> float:
    """1-Wasserstein distance from the empirical distribution of ``values`` to a target
    described by its quantile function ``quantile(p)`` for ``p`` in ``(0, 1)``.

    Zero iff the sorted samples lie exactly on the target's quantiles, and it grows
    smoothly (in the values' own units) as the shape drifts — a good soft *minimization*
    target. Each order statistic ``x_(i)`` is compared to ``quantile((i + 0.5) / n)``;
    the mean absolute gap is the distance. This is the building block for custom fits: a
    downstream distribution only needs to supply its quantile function.
    """
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    return sum(abs(x - quantile((i + 0.5) / n)) for i, x in enumerate(xs)) / n


def _normal_fit(values: list[Any], mu: "int | float", sigma: "int | float") -> float:
    """Distance to ``Normal(mu, sigma)`` — steer a column toward a bell curve."""
    if sigma <= 0:
        raise FandangoValueError(f"normal_fit sigma must be > 0, got {sigma!r}.")
    return wasserstein_fit(values, NormalDist(float(mu), float(sigma)).inv_cdf)


def _lognormal_fit(values: list[Any], mu: "int | float", sigma: "int | float") -> float:
    """Distance to ``LogNormal(mu, sigma)`` — a right-skewed positive column (``mu`` and
    ``sigma`` are the mean/stddev of the underlying normal, i.e. of ``log(value)``)."""
    if sigma <= 0:
        raise FandangoValueError(f"lognormal_fit sigma must be > 0, got {sigma!r}.")
    nd = NormalDist(float(mu), float(sigma))
    return wasserstein_fit(values, lambda p: math.exp(nd.inv_cdf(p)))


def _uniform_fit(values: list[Any], lo: "int | float", hi: "int | float") -> float:
    """Distance to a continuous ``Uniform(lo, hi)`` — flatten a column across a range."""
    if hi <= lo:
        raise FandangoValueError(f"uniform_fit needs lo < hi, got ({lo!r}, {hi!r}).")
    return wasserstein_fit(values, lambda p: lo + p * (hi - lo))


def _exponential_fit(values: list[Any], rate: "int | float") -> float:
    """Distance to ``Exponential(rate)`` (mean ``1/rate``) — a decaying positive column."""
    if rate <= 0:
        raise FandangoValueError(f"exponential_fit rate must be > 0, got {rate!r}.")
    return wasserstein_fit(values, lambda p: -math.log1p(-p) / rate)


# name -> reducer. A reducer takes the list of per-tree values, plus any target
# parameters declared in REDUCER_TARGET_ARITY (evaluated from trailing literal args).
# The built-in fits are deliberately a small, stdlib-only sample; downstream code adds
# more distributions with `register_reducer` (see below) rather than editing this dict.
REDUCERS: dict[str, Callable[..., float]] = {
    "mean": _mean,
    "stddev": _stddev,
    "fraction": _fraction,
    "distinct_count": _distinct_count,
    "count": _count,
    "correlation": _correlation,
    "normal_fit": _normal_fit,
    "lognormal_fit": _lognormal_fit,
    "uniform_fit": _uniform_fit,
    "exponential_fit": _exponential_fit,
}

# How many trailing literal target parameters each reducer expects after the generator,
# e.g. normal_fit(<inner> for x in population, mu, sigma) -> 2. Absent means 0.
REDUCER_TARGET_ARITY: dict[str, int] = {
    "normal_fit": 2,
    "lognormal_fit": 2,
    "uniform_fit": 2,
    "exponential_fit": 1,
}


# --------------------------------------------------------------------------- #
# Marginal companions
# --------------------------------------------------------------------------- #
# A marginal companion answers, cheaply and per value, "how does the aggregate move if this
# value is removed?" -- i.e. it returns the removal influence ``Δ_v = agg_without_v - agg``
# for each value, in the same order as ``values``. This is the O(N) analytic linearization of
# leave-one-out: the ``marginal`` attribution mode evaluates the *same* outer expression as
# ``loo`` but at ``agg + Δ`` instead of re-aggregating the whole population. Companions never
# see the outer expression or the optimization goal -- that sign is applied generically in
# PopulationValue. A reducer with no companion simply falls back to ``loo``.
def _mean_marginal(values: list[Any]) -> list[float]:
    n = len(values)
    if n < 2:
        return [0.0] * n
    total = float(sum(values))
    agg = total / n
    return [(total - v) / (n - 1) - agg for v in values]


def _stddev_marginal(values: list[Any]) -> list[float]:
    n = len(values)
    if n < 2:
        return [0.0] * n
    s = float(sum(values))
    ss = float(sum(v * v for v in values))
    agg = math.sqrt(max(0.0, ss / n - (s / n) ** 2))
    out: list[float] = []
    for v in values:
        if n - 1 < 2:  # _stddev returns 0 for fewer than two values
            agg_v = 0.0
        else:
            mean_v = (s - v) / (n - 1)
            var_v = (ss - v * v) / (n - 1) - mean_v * mean_v
            agg_v = math.sqrt(max(0.0, var_v))
        out.append(agg_v - agg)
    return out


def _fraction_marginal(values: list[Any]) -> list[float]:
    n = len(values)
    if n < 2:
        return [0.0] * n
    truthy = sum(1 for v in values if v)
    agg = truthy / n
    return [(truthy - (1 if v else 0)) / (n - 1) - agg for v in values]


def _count_marginal(values: list[Any]) -> list[float]:
    # Removing any value drops the count by exactly one, so every value has the same
    # influence: count objectives cannot discriminate individuals (expected, harmless).
    return [-1.0] * len(values)


def _distinct_count_marginal(values: list[Any]) -> list[float]:
    counts = Counter(values)
    # Distinct count drops by one only when the removed value was unique in the pool.
    return [-1.0 if counts[v] == 1 else 0.0 for v in values]


def wasserstein_marginal(
    values: list[Any], quantile: Callable[[float], float]
) -> list[float]:
    """Removal influence of each value on :func:`wasserstein_fit` with target ``quantile``.

    The fit is the mean gap ``(1/n) Σ |x_(k) - quantile((k+0.5)/n)|`` over the sorted
    samples. Dropping the sample with gap ``d`` removes that term, so
    ``Δ ≈ (agg - d) / (n - 1)``: a sample placed worse than average (``d > agg``) has a
    negative influence (removing it lowers the distance -> it is penalised under
    ``minimizing``); a well-placed sample has a positive one. This drops the second-order
    requantisation term (the remaining samples renumber), which is exactly why it is a
    sharper, lower-noise signal than yanking a whole tree via ``loo``. Depends only on the
    quantile function, so every distributional fit -- built-in or downstream-registered --
    gets its gradient from the same information it already supplies.
    """
    n = len(values)
    if n < 2:
        return [0.0] * n
    order = sorted(range(n), key=lambda i: float(values[i]))
    gaps = [0.0] * n
    total = 0.0
    for rank, i in enumerate(order):
        d = abs(float(values[i]) - quantile((rank + 0.5) / n))
        gaps[i] = d
        total += d
    agg = total / n
    return [(agg - gaps[i]) / (n - 1) for i in range(n)]


def _normal_marginal(values: list[Any], mu: "int | float", sigma: "int | float") -> list[float]:
    return wasserstein_marginal(values, NormalDist(float(mu), float(sigma)).inv_cdf)


def _lognormal_marginal(
    values: list[Any], mu: "int | float", sigma: "int | float"
) -> list[float]:
    nd = NormalDist(float(mu), float(sigma))
    return wasserstein_marginal(values, lambda p: math.exp(nd.inv_cdf(p)))


def _uniform_fit_marginal(values: list[Any], lo: "int | float", hi: "int | float") -> list[float]:
    return wasserstein_marginal(values, lambda p: lo + p * (hi - lo))


def _exponential_marginal(values: list[Any], rate: "int | float") -> list[float]:
    return wasserstein_marginal(values, lambda p: -math.log1p(-p) / rate)


# name -> marginal companion. A reducer absent from this dict falls back to loo attribution.
# `correlation` is deliberately omitted for now (its per-pair companion is a follow-up; see
# PLAN-marginal-attribution.md §3, §11).
REDUCER_MARGINALS: dict[str, Callable[..., list[float]]] = {
    "mean": _mean_marginal,
    "stddev": _stddev_marginal,
    "fraction": _fraction_marginal,
    "count": _count_marginal,
    "distinct_count": _distinct_count_marginal,
    "normal_fit": _normal_marginal,
    "lognormal_fit": _lognormal_marginal,
    "uniform_fit": _uniform_fit_marginal,
    "exponential_fit": _exponential_marginal,
}


def register_reducer(
    name: str,
    reducer: Callable[..., float],
    *,
    target_arity: int = 0,
    marginal: Optional[Callable[..., list[float]]] = None,
) -> None:
    """Register a population reducer so it can be used in an objective by ``name``.

    This is the supported extension point: a downstream package supplies the
    distributions it needs instead of this repo trying to enumerate them. ``reducer`` is
    called as ``reducer(values, *target_params)`` where ``values`` is the flat list of
    per-tree inner values and ``target_params`` are the ``target_arity`` trailing literal
    arguments from the objective; it must return a float (lower = better under
    ``minimizing``). For a distributional fit, build it from :func:`wasserstein_fit` and a
    quantile function::

        from scipy.stats import gamma
        from fandango.constraints.population import register_reducer, wasserstein_fit

        register_reducer(
            "gamma_fit",
            lambda values, a, scale: wasserstein_fit(
                values, lambda p: gamma.ppf(p, a, scale=scale)
            ),
            target_arity=2,
        )
        # then, in a .fan spec parsed *after* this call:
        #   minimizing gamma_fit([int(<age>) for x in population], 2.0, 10.0)

    Pass ``marginal`` to enable the sharper, cheaper ``marginal`` attribution mode for this
    reducer: a callable ``marginal(values, *target_params) -> list[float]`` returning the
    removal influence ``agg_without_value - agg`` per value (see :func:`wasserstein_marginal`
    and the built-ins). It is optional -- omit it and objectives using this reducer fall back
    to ``loo`` attribution. For the common "I have a quantile function" case, prefer
    :func:`register_distribution_fit`, which wires both the reducer and its marginal from a
    single quantile.

    Registration mutates the process-wide registry, so it must run **before** the spec
    that references ``name`` is parsed. Re-registering an existing name overrides it,
    which lets a downstream provide a better-backed implementation of a built-in.
    """
    if not name.isidentifier():
        raise FandangoValueError(
            f"Reducer name {name!r} must be a valid identifier (it is used as a call "
            f"in objective expressions)."
        )
    if target_arity < 0:
        raise FandangoValueError(f"target_arity must be >= 0, got {target_arity}.")
    if name in REDUCERS:
        LOGGER.info(f"Overriding existing population reducer {name!r}.")
    REDUCERS[name] = reducer
    if target_arity:
        REDUCER_TARGET_ARITY[name] = target_arity
    else:
        REDUCER_TARGET_ARITY.pop(name, None)
    # Keep the marginal registry in lockstep: a re-registration without a marginal must clear
    # any stale companion so the reducer cleanly falls back to loo instead of pairing a new
    # reducer with an old, mismatched gradient.
    if marginal is not None:
        REDUCER_MARGINALS[name] = marginal
    else:
        REDUCER_MARGINALS.pop(name, None)


def register_distribution_fit(
    name: str,
    quantile_factory: Callable[..., Callable[[float], float]],
    *,
    target_arity: int,
) -> None:
    """Register a distributional-fit reducer *and* its marginal from one quantile function.

    ``quantile_factory(*target_params)`` returns the target distribution's quantile function
    ``quantile(p)`` for ``p`` in ``(0, 1)``. Both the :func:`wasserstein_fit` reducer and the
    matching :func:`wasserstein_marginal` gradient are built from it, so a downstream
    distribution supplies *only* a quantile and gets a soft fit objective plus a sharp
    ``marginal`` gradient for free::

        from scipy.stats import gamma
        from fandango.constraints.population import register_distribution_fit

        register_distribution_fit(
            "gamma_fit",
            lambda a, scale: lambda p: gamma.ppf(p, a, scale=scale),
            target_arity=2,
        )
        # then, in a .fan spec parsed *after* this call:
        #   minimizing gamma_fit([int(<age>) for x in population], 2.0, 10.0)
    """
    register_reducer(
        name,
        lambda values, *params: wasserstein_fit(values, quantile_factory(*params)),
        target_arity=target_arity,
        marginal=lambda values, *params: wasserstein_marginal(
            values, quantile_factory(*params)
        ),
    )


# --------------------------------------------------------------------------- #
# Convert-time parsing of a population aggregate
# --------------------------------------------------------------------------- #
@dataclass
class PopulationAggregate:
    """The decomposition of a population objective expression.

    ``outer_expression`` is the original expression with the ``reducer(... for x in
    population)`` call replaced by the reserved :data:`AGGREGATE_PLACEHOLDER` name, so
    that ``eval(outer_expression, {AGGREGATE_PLACEHOLDER: agg})`` reproduces the score.
    ``inner_expression`` is the generator element, evaluated *per tree* through the
    normal search machinery. ``inner_searches`` is the subset of the objective's
    searches referenced by ``inner_expression``.
    """

    reducer_name: str
    inner_expression: str
    outer_expression: str
    loop_var: str
    inner_searches: dict[str, NonTerminalSearch] = field(default_factory=dict)
    # Literal target parameters following the generator, e.g. the (mu, sigma) of
    # normal_fit(<inner> for x in population, 30, 5).
    reducer_args: list[Any] = field(default_factory=list)


# Reflection of a comparison operator under an operand swap: ``a OP b`` is equivalent to
# ``b _REFLECTED_OPERATOR[OP] a``. Used to canonicalize a population-on-the-right comparison
# (``50 <= distinct_count(...)``) into aggregate-on-the-left form. NOTE: this is *not*
# ``Comparison.invert()``, which is logical negation (``==`` -> ``!=``).
_REFLECTED_OPERATOR: dict[Comparison, Comparison] = {
    Comparison.EQUAL: Comparison.EQUAL,
    Comparison.NOT_EQUAL: Comparison.NOT_EQUAL,
    Comparison.GREATER: Comparison.LESS,
    Comparison.GREATER_EQUAL: Comparison.LESS_EQUAL,
    Comparison.LESS: Comparison.GREATER,
    Comparison.LESS_EQUAL: Comparison.GREATER_EQUAL,
}


@dataclass
class PopulationRequirement:
    """A hard, population-scoped ``where``: over the emitted batch of N individuals, the
    aggregate must satisfy ``operator(aggregate, bound)``.

    Built by :func:`try_parse_population_requirement` from a single top-level
    ``ComparisonConstraint`` one side of which is a ``reducer(<inner> for x in population)``
    aggregate. The operator is *canonicalized* so the aggregate is always the left operand:
    a population-on-the-right comparison (``50 <= distinct_count(...)``) is reflected to the
    equivalent aggregate-on-the-left form (``distinct_count(...) >= 50``).

    ``bound`` is the user's literal target verbatim. Any batch-size-dependent snapping of an
    exact target -- e.g. ``fraction(...) == 0.30`` rounding to ``round(0.30 * N) / N`` -- is
    deferred to the sampler, where N is known; the detector only records the requirement.
    """

    aggregate: PopulationAggregate
    operator: Comparison
    bound: Any


class _InnerValue(Value):
    """Concrete per-tree ``Value`` (``Value`` itself is abstract) used to evaluate the
    generator element against a single individual and read its raw ``.values``."""

    def format_as_spec(self) -> str:
        return self.expression

    def raw_values(
        self,
        tree: DerivationTree,
        scope: Optional[dict[Any, DerivationTree]] = None,
        local_variables: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        """The per-combination results of the inner expression as raw Python objects.

        Mirrors :meth:`Value.fitness`'s evaluation loop but returns the results directly
        instead of wrapping them in a numeric :class:`ValueFitness` — a *joint* inner
        expression yields tuples (e.g. ``(age, income)``), which are not fitness numbers.
        """
        results: list[Any] = []
        for combination in self.combinations(tree, scope):
            local_vars = self.local_variables.copy()
            if local_variables:
                local_vars.update(local_variables)
            local_vars.update(
                {name: container.evaluate() for name, container in combination}
            )
            results.append(eval(self.expression, self.global_variables, local_vars))
        return results


class _AggregateReplacer(ast.NodeTransformer):
    def __init__(self, target: ast.AST):
        self._target = target

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if node is self._target:
            return ast.copy_location(ast.Name(id=AGGREGATE_PLACEHOLDER, ctx=ast.Load()), node)
        return self.generic_visit(node)


def _references_population(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Name) and n.id == POPULATION_BINDER for n in ast.walk(node)
    )


def _is_population_reducer_call(node: ast.AST) -> bool:
    """A call ``reducer(<genexp/listcomp> for x in population, *target_params)``.

    The generator must be the first argument; any further positional arguments are the
    reducer's literal target parameters (e.g. the mu/sigma of ``normal_fit``)."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in REDUCERS
        and len(node.args) >= 1
        and not node.keywords
        and isinstance(node.args[0], (ast.GeneratorExp, ast.ListComp))
    ):
        return False
    comp = node.args[0]
    return (
        len(comp.generators) == 1
        and isinstance(comp.generators[0].iter, ast.Name)
        and comp.generators[0].iter.id == POPULATION_BINDER
    )


def try_parse_population_aggregate(
    expression: str,
    searches: Optional[dict[str, NonTerminalSearch]] = None,
) -> Optional[PopulationAggregate]:
    """Decompose ``expression`` into a :class:`PopulationAggregate`, or ``None``.

    Returns ``None`` when the expression does not reference the reserved
    ``population`` binder (i.e. it is an ordinary soft value). Raises
    :class:`FandangoValueError` when ``population`` *is* referenced but the shape is
    unsupported, so that misuse surfaces clearly rather than silently degrading to a
    per-tree soft value.

    ``expression`` is the *post-substitution* string produced by ``visitExpr`` — i.e.
    non-terminals such as ``<age>`` have already been replaced by search placeholders,
    and those placeholders are the keys of ``searches``.
    """
    searches = searches or {}
    try:
        module = ast.parse(expression, mode="eval")
    except SyntaxError as e:  # pragma: no cover - expression already came from unparse
        raise FandangoValueError(
            f"Could not parse population objective {expression!r}: {e}"
        )
    root = module.body

    if not _references_population(root):
        return None

    reducer_calls = [n for n in ast.walk(root) if _is_population_reducer_call(n)]
    if not reducer_calls:
        raise FandangoValueError(
            f"'{POPULATION_BINDER}' may only be used as the iterable of a reducer, "
            f"e.g. mean(<inner> for x in {POPULATION_BINDER}); got {expression!r}. "
            f"Supported reducers: {', '.join(sorted(REDUCERS))}."
        )
    if len(reducer_calls) > 1:
        raise FandangoValueError(
            f"Only one population aggregate per objective is supported; "
            f"{expression!r} contains {len(reducer_calls)}."
        )

    call = reducer_calls[0]
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    reducer_name = call.func.id

    # Target parameters (e.g. the mu/sigma of normal_fit) follow the generator as
    # literal positional args; evaluate them now so the reducer stays a pure list->float.
    try:
        reducer_args = [ast.literal_eval(a) for a in call.args[1:]]
    except (ValueError, SyntaxError):
        raise FandangoValueError(
            f"Target parameters of '{reducer_name}' must be literals, e.g. "
            f"normal_fit(<inner> for x in {POPULATION_BINDER}, 30, 5); got {expression!r}."
        )
    expected_arity = REDUCER_TARGET_ARITY.get(reducer_name, 0)
    if len(reducer_args) != expected_arity:
        raise FandangoValueError(
            f"'{reducer_name}' takes {expected_arity} target parameter(s), "
            f"got {len(reducer_args)}: {expression!r}."
        )

    comp = call.args[0]
    assert isinstance(comp, (ast.GeneratorExp, ast.ListComp))
    generator = comp.generators[0]

    if generator.ifs:
        raise FandangoValueError(
            f"Filters ('if') in a population aggregate are not supported yet: {expression!r}"
        )
    if generator.is_async:
        raise FandangoValueError(
            f"'async for' is not supported in a population aggregate: {expression!r}"
        )
    if not isinstance(generator.target, ast.Name):
        raise FandangoValueError(
            f"The population loop target must be a single variable: {expression!r}"
        )
    loop_var = generator.target.id

    elt = comp.elt
    if _references_population(elt):
        raise FandangoValueError(
            f"Nested references to '{POPULATION_BINDER}' are not supported: {expression!r}"
        )

    # Extract the inner element (and its searches) *before* mutating the tree.
    inner_expression = ast.unparse(elt)
    referenced = {n.id for n in ast.walk(elt) if isinstance(n, ast.Name)}
    inner_searches = {k: v for k, v in searches.items() if k in referenced}

    # Replace the reducer call in-place on the *same* tree so node identity matches.
    outer_tree = _AggregateReplacer(call).visit(module)
    ast.fix_missing_locations(outer_tree)
    outer_expression = ast.unparse(outer_tree.body)

    return PopulationAggregate(
        reducer_name=reducer_name,
        inner_expression=inner_expression,
        outer_expression=outer_expression,
        loop_var=loop_var,
        inner_searches=inner_searches,
        reducer_args=reducer_args,
    )


def _parse_requirement_bound(
    source: str,
    searches: dict[str, NonTerminalSearch],
    reducer_name: str,
) -> Any:
    """The comparison target of a population requirement must be a compile-time numeric
    literal: the batch is *constructed* toward a fixed number, so the bound cannot depend on
    per-tree grammar values. A target that references a symbol (``searches`` non-empty) or is
    otherwise non-literal / non-numeric is rejected with a clear message."""
    example = f"{reducer_name}(<inner> for x in {POPULATION_BINDER}) == 0.30"
    if searches:
        raise FandangoValueError(
            f"The target of a population requirement must be a constant, not a grammar "
            f"symbol: got {source!r}. Write a fixed number, e.g. '{example}'."
        )
    try:
        bound = ast.literal_eval(source)
    except (ValueError, SyntaxError, TypeError):
        raise FandangoValueError(
            f"The target of a population requirement must be a literal number, e.g. "
            f"'{example}'; got {source!r}."
        )
    if isinstance(bound, bool) or not isinstance(bound, (int, float)):
        raise FandangoValueError(
            f"The target of a population requirement must be a number, e.g. '{example}'; "
            f"got {bound!r}."
        )
    return bound


def try_parse_population_requirement(
    constraint: Constraint,
) -> Optional[PopulationRequirement]:
    """Detect a hard population-scoped ``where`` in a freshly-built constraint.

    Returns a :class:`PopulationRequirement` when ``constraint`` is a single
    ``ComparisonConstraint`` with a ``reducer(<inner> for x in population)`` aggregate on one
    side; returns ``None`` for an ordinary per-tree constraint (no ``population`` reference).
    Raises :class:`FandangoValueError` when ``population`` *is* referenced but the shape is
    unsupported in v1 -- a compound ``and``/``or``, a bare aggregate with no comparison, both
    sides aggregating, or a non-literal target -- so misuse surfaces clearly instead of
    silently degrading to a per-tree constraint (where ``population`` is unbound and crashes).

    Detector only: the caller (``visitConstraint``'s ``where`` branch) routes the result to
    the sampler layer rather than the per-tree evaluator. Reaching into the constraint's
    ``_left``/``_right`` operands is deliberate -- they carry the post-substitution source and
    split searches that :func:`try_parse_population_aggregate` already consumes.
    """
    if isinstance(constraint, ComparisonConstraint):
        left = try_parse_population_aggregate(
            constraint._left, constraint._left_searches
        )
        right = try_parse_population_aggregate(
            constraint._right, constraint._right_searches
        )
        if left is None and right is None:
            return None  # ordinary per-tree `where`
        if left is not None and right is not None:
            raise FandangoValueError(
                f"'{POPULATION_BINDER}' may appear on only one side of a population "
                f"requirement; got it on both in {constraint.format_as_spec()!r}."
            )
        if left is not None:
            aggregate = left
            operator = constraint._operator
            bound_source, bound_searches = constraint._right, constraint._right_searches
        else:
            assert right is not None
            aggregate = right
            operator = _REFLECTED_OPERATOR[constraint._operator]
            bound_source, bound_searches = constraint._left, constraint._left_searches
        bound = _parse_requirement_bound(
            bound_source, bound_searches, aggregate.reducer_name
        )
        return PopulationRequirement(
            aggregate=aggregate, operator=operator, bound=bound
        )

    # Any non-comparison shape that mentions the reserved binder is an unsupported v1
    # population `where` (compound `and`/`or`, a bare aggregate, ...). Reject it clearly
    # rather than let it reach the per-tree Evaluator, where `population` is unbound. The
    # binder can only appear as `for x in population` in the reconstructed source, so a
    # word-boundary match on `format_as_spec()` is unambiguous (`population` is reserved).
    if re.search(rf"\b{POPULATION_BINDER}\b", constraint.format_as_spec()):
        raise FandangoValueError(
            f"A population 'where' must be a single comparison, e.g. "
            f"fraction(<inner> for x in {POPULATION_BINDER}) == 0.30; got "
            f"{constraint.format_as_spec()!r}. Split a compound requirement into separate "
            f"'where' lines, and give a bare aggregate a comparison and target."
        )
    return None


# --------------------------------------------------------------------------- #
# PopulationValue
# --------------------------------------------------------------------------- #
class PopulationValue(SoftValue):
    """A soft objective evaluated over the whole population, not one tree.

    Subclasses :class:`SoftValue` so every existing ``isinstance(x, SoftValue)`` check
    and ``list[Constraint | SoftValue]`` union keeps working unchanged; the evaluator
    must route :class:`PopulationValue` *before* the generic soft branch (separate step).

    The inherited ``tdigest`` normalizes the *aggregate* outer score across generations,
    exactly as it normalizes a per-tree soft value.
    """

    ATTRIBUTIONS = ("uniform", "loo", "marginal")

    def __init__(
        self,
        optimization_goal: str,
        expression: str,
        *,
        aggregate: PopulationAggregate,
        attribution: str = "loo",
        local_variables: Optional[dict[str, Any]] = None,
        global_variables: Optional[dict[str, Any]] = None,
    ):
        # self.searches are the *inner* searches, so combinations()/get_access_points()
        # and SoftValue.format_as_spec() operate on the per-tree part.
        super().__init__(
            optimization_goal,
            expression,
            searches=aggregate.inner_searches,
            local_variables=local_variables,
            global_variables=global_variables,
        )
        if attribution not in self.ATTRIBUTIONS:
            raise FandangoValueError(
                f"Unknown attribution {attribution!r}; expected one of {self.ATTRIBUTIONS}."
            )
        self.aggregate = aggregate
        self.attribution = attribution
        # A plain per-tree Value used to evaluate the inner element expression against
        # each individual. Reuses all normal search/scope resolution — no new search code.
        self._inner_value = _InnerValue(
            aggregate.inner_expression,
            searches=aggregate.inner_searches,
            local_variables=self.local_variables,
            global_variables=self.global_variables,
        )
        # Row-scoping (prototype). When the inner expression combines >= 2 distinct
        # non-terminals it is a *joint* objective: the fields must be paired within a row,
        # not cross-producted over the whole tree (which would destroy every joint
        # statistic — corr(x, y) collapses to 0). We then evaluate the inner expression
        # against each *row* subtree instead. The row non-terminal is the tightest one
        # whose every instance holds exactly one match of each field; it is inferred from
        # the first population and cached in `_row_symbol`.
        self._target_symbols = sorted(
            {
                str(nt)
                for s in aggregate.inner_searches.values()
                for nt in s.get_access_points()
            }
        )
        self._row_scoped = len(self._target_symbols) >= 2
        self._row_symbol: Optional[Any] = None

    def fitness(self, tree, scope=None, local_variables=None) -> ValueFitness:  # type: ignore[override]
        """Neutral per-tree contract.

        A population objective has no meaningful single-tree fitness. We return a
        neutral, empty :class:`ValueFitness` (rather than raising) so that any code path
        which iterates *all* constraints and calls ``fitness(tree)`` stays harmless. The
        real work happens in :meth:`evaluate_population`.
        """
        return ValueFitness([])

    # -- per-tree inner values --------------------------------------------- #
    def _inner_values_per_tree(
        self, population: list[DerivationTree]
    ) -> list[list[Any]]:
        """The raw inner values for each individual (a tree may yield several).

        For a marginal objective the inner expression is evaluated once against the whole
        tree. For a *joint* (row-scoped) objective it is evaluated once per row subtree,
        so multiple fields stay paired; see :meth:`_infer_row_symbol`.
        """
        per_tree: list[list[Any]] = []
        for tree in population:
            if self._row_scoped:
                per_tree.append(self._row_scoped_values(tree))
            else:
                extra = {self.aggregate.loop_var: tree}
                values = self._inner_value.fitness(tree, local_variables=extra).values
                per_tree.append(list(values))
        return per_tree

    def _row_scoped_values(self, tree: DerivationTree) -> list[Any]:
        """One inner value per row of ``tree``, evaluating the inner expression against
        each row subtree so its fields are paired row-wise instead of cross-producted."""
        if self._row_symbol is None:
            self._row_symbol = self._infer_row_symbol(tree)
        if self._row_symbol is None:
            # No non-terminal partitions the fields one-per-row; we cannot align them, so
            # emit no signal for this tree rather than silently cross-producting.
            LOGGER.warning(
                f"Could not infer a row non-terminal pairing {self._target_symbols} for "
                f"joint objective {self.format_as_spec()}; skipping this individual."
            )
            return []
        row_values: list[Any] = []
        for row in tree.find_subtrees(self._row_symbol):
            extra = {self.aggregate.loop_var: row}
            row_values.extend(self._inner_value.raw_values(row, local_variables=extra))
        return row_values

    def _infer_row_symbol(self, tree: DerivationTree) -> Optional[Any]:
        """The tightest non-terminal whose every subtree holds exactly one match of each
        target field — i.e. one table row. Returns ``None`` if none qualifies."""
        searches = list(self.aggregate.inner_searches.values())
        best: Optional[tuple[int, Any]] = None
        seen: set[str] = set()
        stack = [tree]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            if not node.symbol.is_non_terminal or str(node.symbol) in seen:
                continue
            seen.add(str(node.symbol))
            rows = list(tree.find_subtrees(node.symbol))
            if not rows:
                continue
            if all(len(s.find(row)) == 1 for row in rows for s in searches):
                # Tightest = most rows (each holding exactly one of every field).
                if best is None or len(rows) > best[0]:
                    best = (len(rows), node.symbol)
        return best[1] if best else None

    # -- aggregate -> outer score ------------------------------------------ #
    def _aggregate(self, values: Iterable[Any]) -> float:
        """Run the reducer over ``values`` to the single aggregate scalar."""
        return REDUCERS[self.aggregate.reducer_name](
            list(values), *self.aggregate.reducer_args
        )

    def _outer_from_aggregate(self, aggregate: float) -> float:
        """Evaluate the outer expression with the aggregate substituted in.

        Split out from :meth:`_outer_score` so ``marginal`` attribution can score a
        linearly-perturbed aggregate (``agg + Δ``) without re-running the reducer.
        """
        return float(
            eval(
                self.aggregate.outer_expression,
                self.global_variables,
                {AGGREGATE_PLACEHOLDER: aggregate},
            )
        )

    def _outer_score(self, values: Iterable[Any]) -> float:
        return self._outer_from_aggregate(self._aggregate(values))

    def _goal_adjusted(self, outer_score: float, *, update: bool) -> float:
        """Normalize an outer score to [0, 1] where higher is always better."""
        if update:
            self.tdigest.update(outer_score)
        normalized = self.tdigest.score(outer_score)
        return normalized if self.optimization_goal == "max" else 1 - normalized

    # -- attribution: aggregate score -> per-tree reward ------------------- #
    def _reward_sign(self, outer_perturbed: float, outer_full: float) -> float:
        """A positive reward means "this tree pulled the aggregate toward the goal".

        For ``min`` the objective should go *up* when a helpful tree is removed; for
        ``max`` it should go *down*. ``outer_perturbed`` is the outer score with the tree's
        contribution removed (exactly, for ``loo``; linearly approximated, for ``marginal``).
        """
        if self.optimization_goal == "min":
            return outer_perturbed - outer_full
        return outer_full - outer_perturbed

    def _loo_rewards(
        self, per_tree: list[list[Any]], outer_full: float
    ) -> list[float]:
        """Leave-one-out: re-score the objective over the population minus each tree."""
        rewards: list[float] = []
        for i in range(len(per_tree)):
            rest = [v for j, values in enumerate(per_tree) if j != i for v in values]
            if not rest:
                rewards.append(0.0)
                continue
            try:
                outer_loo = self._outer_score(rest)
            except Exception:
                outer_loo = outer_full
            rewards.append(self._reward_sign(outer_loo, outer_full))
        return rewards

    def _marginal_rewards(
        self,
        per_tree: list[list[Any]],
        all_values: list[Any],
        agg_full: float,
        outer_full: float,
    ) -> list[float]:
        """The O(N) linearization of :meth:`_loo_rewards`.

        Ask the reducer's marginal companion for each value's removal influence
        ``Δ_v = agg_without_v - agg``, sum those over each tree, and score the outer
        expression once per tree at ``agg + Σ Δ_v`` — the first-order approximation of the
        leave-one-tree-out aggregate, with no re-aggregation. Falls back to ``loo`` when the
        reducer has no companion (e.g. ``correlation``) or the population is too small.
        """
        marginal = REDUCER_MARGINALS.get(self.aggregate.reducer_name)
        if marginal is None or len(all_values) < 2:
            LOGGER.info(
                f"No marginal companion for reducer {self.aggregate.reducer_name!r} "
                f"(or population too small); falling back to loo attribution."
            )
            return self._loo_rewards(per_tree, outer_full)
        try:
            deltas = marginal(all_values, *self.aggregate.reducer_args)
        except Exception as e:
            LOGGER.error(
                f"Marginal companion for {self.aggregate.reducer_name!r} failed ({e}); "
                f"falling back to loo attribution."
            )
            return self._loo_rewards(per_tree, outer_full)

        rewards: list[float] = []
        cursor = 0
        for values in per_tree:
            k = len(values)
            if k == 0:
                rewards.append(0.0)
                continue
            delta_tree = sum(deltas[cursor : cursor + k])
            cursor += k
            try:
                outer_tree = self._outer_from_aggregate(agg_full + delta_tree)
            except Exception:
                outer_tree = outer_full
            rewards.append(self._reward_sign(outer_tree, outer_full))
        return rewards

    # -- public entry point ------------------------------------------------ #
    def evaluate_population(self, population: list[DerivationTree]) -> list[float]:
        """A per-individual bonus vector in ``[0, 1]`` for the given population."""
        n = len(population)
        if n == 0:
            return []

        per_tree = self._inner_values_per_tree(population)
        all_values = [v for values in per_tree for v in values]
        if not all_values:
            # Nothing to aggregate (e.g. constraint not yet satisfiable) — no signal.
            return [0.0] * n

        try:
            agg_full = self._aggregate(all_values)
            outer_full = self._outer_from_aggregate(agg_full)
        except Exception as e:  # keep the GA alive on a bad user expression
            LOGGER.error(
                f"Error evaluating population objective {self.format_as_spec()}: {e}"
            )
            return [0.0] * n

        base = self._goal_adjusted(outer_full, update=True)

        if self.attribution == "uniform":
            return [base] * n

        # Reward each individual by how much *including* it moves the objective toward its
        # goal. `loo` measures this exactly by re-aggregation; `marginal` approximates it
        # analytically in O(N). The bonus is additive — `0.5*base + 0.5*reward` — not
        # multiplicative, so a cold `tdigest` (base ~= 0 for the first few generations) does
        # not wipe out the per-individual selection gradient exactly when the GA needs it to
        # start moving. Rewards are min-max normalized across the population into [0, 1]
        # (0.5 for everyone when there is no spread).
        #
        # NOTE: min-max normalization amplifies arbitrarily small reward differences to the
        # full range; smarter scaling is a tuning lever kept separate so `loo` vs `marginal`
        # A/Bs change exactly one variable (see PLAN-marginal-attribution.md §5, §11).
        if self.attribution == "marginal":
            rewards = self._marginal_rewards(per_tree, all_values, agg_full, outer_full)
        else:
            rewards = self._loo_rewards(per_tree, outer_full)

        lo, hi = min(rewards), max(rewards)
        if hi - lo < 1e-12:
            norm = [0.5] * n
        else:
            norm = [(r - lo) / (hi - lo) for r in rewards]
        return [0.5 * base + 0.5 * r for r in norm]

    def format_as_spec(self) -> str:
        # SoftValue.format_as_spec replaces inner search placeholders in self.expression;
        # that already yields the readable original objective.
        return super().format_as_spec()
