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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Optional

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


def register_reducer(
    name: str,
    reducer: Callable[..., float],
    *,
    target_arity: int = 0,
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


class _InnerValue(Value):
    """Concrete per-tree ``Value`` (``Value`` itself is abstract) used to evaluate the
    generator element against a single individual and read its raw ``.values``."""

    def format_as_spec(self) -> str:
        return self.expression


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

    ATTRIBUTIONS = ("uniform", "loo")

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
        """The raw inner values for each individual (a tree may yield several)."""
        per_tree: list[list[Any]] = []
        for tree in population:
            extra = {self.aggregate.loop_var: tree}
            values = self._inner_value.fitness(tree, local_variables=extra).values
            per_tree.append(list(values))
        return per_tree

    # -- aggregate -> outer score ------------------------------------------ #
    def _outer_score(self, values: Iterable[Any]) -> float:
        values = list(values)
        aggregate = REDUCERS[self.aggregate.reducer_name](
            values, *self.aggregate.reducer_args
        )
        return float(
            eval(
                self.aggregate.outer_expression,
                self.global_variables,
                {AGGREGATE_PLACEHOLDER: aggregate},
            )
        )

    def _goal_adjusted(self, outer_score: float, *, update: bool) -> float:
        """Normalize an outer score to [0, 1] where higher is always better."""
        if update:
            self.tdigest.update(outer_score)
        normalized = self.tdigest.score(outer_score)
        return normalized if self.optimization_goal == "max" else 1 - normalized

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
            outer_full = self._outer_score(all_values)
        except Exception as e:  # keep the GA alive on a bad user expression
            LOGGER.error(
                f"Error evaluating population objective {self.format_as_spec()}: {e}"
            )
            return [0.0] * n

        base = self._goal_adjusted(outer_full, update=True)

        if self.attribution == "uniform":
            return [base] * n

        # Leave-one-out: reward each individual by how much *including* it moves the
        # objective toward its goal. The bonus is additive — `0.5*base + 0.5*reward` —
        # not multiplicative, so a cold `tdigest` (base ~= 0 for the first few
        # generations) does not wipe out the per-individual selection gradient exactly
        # when the GA needs it to start moving. Rewards are min-max normalized across
        # the population into [0, 1] (0.5 for everyone when there is no spread).
        #
        # NOTE: min-max normalization amplifies arbitrarily small reward differences to
        # the full range; smarter scaling is a tuning lever to revisit when benchmarking
        # attribution modes (see PLAN §2, §6).
        rewards: list[float] = []
        for i in range(n):
            rest = [v for j, values in enumerate(per_tree) if j != i for v in values]
            if not rest:
                rewards.append(0.0)
                continue
            try:
                outer_loo = self._outer_score(rest)
            except Exception:
                outer_loo = outer_full
            # For "min" the objective should go *up* when a helpful individual is
            # removed; for "max" it should go *down*. Either way a positive reward means
            # "this individual pulled the aggregate toward the goal".
            if self.optimization_goal == "min":
                rewards.append(outer_loo - outer_full)
            else:
                rewards.append(outer_full - outer_loo)

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
