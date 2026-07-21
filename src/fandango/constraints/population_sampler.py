"""Mechanism B: *constructing* a batch that satisfies a hard population requirement.

Where the GA steers a per-tree fitness (best-effort), a hard population ``where`` is a
guarantee over the whole emitted batch of N -- something no single individual can satisfy and
that streaming, per-tree evolution cannot express. This sampler sits *above* the GA and
constructs such a batch directly.

v1 covers three construction cases, all requiring exactly one inner value per individual:

- **fraction quota** -- ``fraction(<pred> for x in population) OP p``: buckets fuzzed individuals
  by the boolean predicate and assembles the target count, hitting the fraction exactly (for
  ``==``, snapped to the nearest achievable ``round(p*N)/N``) or the operator's boundary.
- **distinct-value diversity** -- ``distinct_count(<field> for x in population) OP K``: collects
  representatives until the required number of *distinct* field values is reached (``>=``/``>``/
  ``==``) or capped (``<=``/``<``), then fills the batch by reusing those representatives so the
  distinct count lands exactly on target.
- **distributional fit** -- ``normal_fit([<x> for x in population], ...) OP delta`` (``<=``/``<``):
  fuzzes a candidate pool and selects, for each order-statistic slot, the individual whose value is
  nearest the target quantile, so the batch's distribution matches the target within delta.

Multiple population requirements are combined when they target **disjoint fields**: each requirement
plans a per-tree column of source individuals (via the same selection logic), and the chosen field
subtree is grafted into shared skeleton individuals -- disjoint fields mean the grafts don't
interfere, so each requirement's verify gate holds independently. Requirements on the same field, or
whose fields nest, are rejected.

A final verification gate re-checks the assembled batch in every case, and per-tree hard constraints
from the spec are co-enforced -- every constructed individual is drawn to satisfy them (the
candidate source rejection-fuzzes; grafted individuals are re-checked). Everything else raises a
clear error rather than silently degrading: multi-valued inner fields (the grouping-policy case)
are future work.

Coupled `correlation((<x>, <y>) for x in population) OP r` is constructed by pairing the two
fields' fuzzed values and grafting both together into a skeleton so the pair holds per individual.
Inequalities (`>=`/`>`/`<=`/`<`) pair monotonically (or anti-monotonically) to drive toward the
+/-1 extreme; `== r` targets a *specific* correlation via a Gaussian-copula rank pairing, searched
over many draws for the one whose achieved correlation is nearest `r` (exact Pearson equality is
unachievable on discrete values, so the gate is a `correlation_tolerance` band -- the analogue of
the fraction `==` snap-and-warn). Conditional `P(y|x)` (via the umbrella-symbol pattern -- emit the
whole correlated tuple from one `:=` generator) and >2-way coupling are future work.

Every reducer the sampler can construct is a :class:`_RequirementHandler` in ``_BUILTIN_HANDLERS``;
validate/plan/verify all dispatch through ``_handler_for``. A downstream package can add a custom
single-field requirement with :func:`fandango.constraints.population.register_requirement` (a
paired ``check`` + ``sample`` handler), which the sampler adapts and constructs toward by pinning a
column to the handler's drawn values.
"""

import ast
import bisect
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Optional

from fandango.constraints.constraint import Constraint
from fandango.constraints.failing_tree import Comparison
from fandango.constraints.population import (
    REDUCERS,
    REQUIREMENT_HANDLERS,
    PopulationRequirement,
    RequirementHandler,
    _InnerValue,
    grouping_for,
)
from fandango.errors import FandangoValueError
from fandango.language.grammar.grammar import Grammar
from fandango.language.tree import DerivationTree
from fandango.logger import LOGGER
from fandango.statistics.equivalence import discretization_floor, is_delta_below_floor

# Comparison operators the sampler can construct toward. ``!=`` is intentionally excluded: "not
# exactly 30%" / "not exactly K distinct" is an odd guarantee with no natural construction target.
_CONSTRUCTIBLE_OPERATORS = frozenset(
    {
        Comparison.EQUAL,
        Comparison.GREATER_EQUAL,
        Comparison.LESS_EQUAL,
        Comparison.GREATER,
        Comparison.LESS,
    }
)


# Distributional-fit reducers, mapped to a factory that turns their literal target parameters into
# the target distribution's quantile function Q(p). The fit is the mean gap to these quantiles, so
# placing the batch's order statistics on Q((i+0.5)/N) drives the fit toward its discretization
# floor. These mirror the quantiles the fits themselves integrate against in population.py.
# Plain (unannotated) so beartype's claw hook doesn't try to wrap functions that return a bare
# callable -- it chokes on `Callable[[float], float]` in return position here.
def _normal_quantile(mu, sigma):
    return NormalDist(float(mu), float(sigma)).inv_cdf


def _lognormal_quantile(mu, sigma):
    nd = NormalDist(float(mu), float(sigma))
    return lambda p: math.exp(nd.inv_cdf(p))


def _uniform_quantile(lo, hi):
    return lambda p: lo + p * (hi - lo)


def _exponential_quantile(rate):
    return lambda p: -math.log1p(-p) / rate


_DISTRIBUTION_QUANTILES = {
    "normal_fit": _normal_quantile,
    "lognormal_fit": _lognormal_quantile,
    "uniform_fit": _uniform_quantile,
    "exponential_fit": _exponential_quantile,
}


@dataclass(frozen=True)
class _RequirementHandler:
    """How the sampler constructs and verifies one reducer's requirement.

    Collapses what used to be three parallel ``if reducer == ...`` ladders (validate, plan,
    verify) into a single registry entry. ``construct`` and ``verify`` take the sampler
    explicitly (they call its private helpers) so the registry can live at module scope; the
    lambdas resolve the bound methods lazily at call time.

    :param construct: ``(sampler, req, n)`` -> the per-slot source column. Single-field
        reducers return a ``list[DerivationTree]`` (whole carriers); a coupled reducer returns
        a ``list[dict[symbol, carrier]]`` directly.
    :param verify: ``(sampler, req, inner, batch)`` -> None; raises on a failed gate.
    :param allowed_operators: operators this reducer can be constructed toward (a subset of
        ``_CONSTRUCTIBLE_OPERATORS``).
    :param arity: number of grammar fields the inner element reads (1 marginal, 2 coupled).
    :param coupled: whether the reducer jointly constrains several fields of one individual.
    :param operator_hint: appended to the error when an operator is not allowed.
    :param quantile_factory: for distributional fits, ``(*params) -> Q(p)``; else ``None``.
    :param grouping: multiplicity policy for multi-valued fields (P5); ``"pool"`` for built-ins.
    """

    construct: Callable[..., list]
    verify: Callable[..., None]
    allowed_operators: frozenset
    arity: int = 1
    coupled: bool = False
    operator_hint: str = ""
    quantile_factory: Optional[Callable[..., Callable[[float], float]]] = None
    grouping: str = "pool"


# Built-in requirement handlers, keyed by reducer name. Defined here (lambdas resolve the
# sampler's methods lazily) so validate/plan/verify all dispatch through one table. A custom
# handler registered via register_requirement (see population.register_requirement) is looked
# up the same way through `_handler_for`.
_FIT_OPERATORS = frozenset({Comparison.LESS_EQUAL, Comparison.LESS})
_CORRELATION_OPERATORS = frozenset(
    {
        Comparison.EQUAL,
        Comparison.GREATER_EQUAL,
        Comparison.GREATER,
        Comparison.LESS_EQUAL,
        Comparison.LESS,
    }
)

_BUILTIN_HANDLERS: dict[str, _RequirementHandler] = {
    "fraction": _RequirementHandler(
        construct=lambda s, req, n: s._fraction_plan(req, n),
        verify=lambda s, req, inner, batch: s._verify(
            req, inner, batch, s._target_count(req, len(batch))
        ),
        allowed_operators=_CONSTRUCTIBLE_OPERATORS,
    ),
    "distinct_count": _RequirementHandler(
        construct=lambda s, req, n: s._distinct_plan(req, n),
        verify=lambda s, req, inner, batch: s._verify_distinct(req, inner, batch),
        allowed_operators=_CONSTRUCTIBLE_OPERATORS,
    ),
    "correlation": _RequirementHandler(
        construct=lambda s, req, n: s._correlation_plan(req, n),
        verify=lambda s, req, inner, batch: s._verify_correlation(req, inner, batch),
        allowed_operators=_CORRELATION_OPERATORS,
        arity=2,
        coupled=True,
        operator_hint="correlation supports `==` (approximate) and `>=`/`>`/`<=`/`<`",
    ),
    **{
        name: _RequirementHandler(
            construct=lambda s, req, n: s._distribution_plan(req, n),
            verify=lambda s, req, inner, batch: s._verify_fit(req, inner, batch),
            allowed_operators=_FIT_OPERATORS,
            operator_hint=(
                "a distributional fit is a distance to a target; only `<=`/`<` are "
                "meaningful (match within delta)"
            ),
            quantile_factory=factory,
        )
        for name, factory in _DISTRIBUTION_QUANTILES.items()
    },
}


def _adapt_custom_handler(reducer: str, custom: RequirementHandler) -> _RequirementHandler:
    """Wrap a user :class:`RequirementHandler` (value-level ``sample``/``check``) as a sampler
    ``_RequirementHandler`` whose construct pins a column to the drawn values and whose verify
    runs ``check`` (see :meth:`PopulationSampler._sample_value_plan` / ``_verify_custom``)."""
    if custom.allowed_operators is None:
        allowed = _CONSTRUCTIBLE_OPERATORS
    else:
        allowed = frozenset(
            op for op in _CONSTRUCTIBLE_OPERATORS if op.value in custom.allowed_operators
        )
    return _RequirementHandler(
        construct=lambda s, req, n: s._sample_value_plan(req, n),
        verify=lambda s, req, inner, batch: s._verify_custom(req, inner, batch),
        allowed_operators=allowed,
        arity=1,
        coupled=False,
        operator_hint=(
            f"'{reducer}' allows {sorted(op.value for op in allowed)}"
        ),
        grouping=custom.grouping,
    )


def _handler_for(reducer: str) -> Optional[_RequirementHandler]:
    """The handler for ``reducer`` (built-in or custom-registered), or ``None`` if the sampler
    cannot construct it. A custom handler is constructible only when it supplies a ``sample``."""
    if reducer in _BUILTIN_HANDLERS:
        return _BUILTIN_HANDLERS[reducer]
    custom = REQUIREMENT_HANDLERS.get(reducer)
    if custom is not None and custom.sample is not None:
        return _adapt_custom_handler(reducer, custom)
    return None


def _requirement_symbols(req: PopulationRequirement) -> list[str]:
    """The (sorted, deduplicated) set of grammar symbols a requirement's inner element reads. One
    symbol is the marginal case; two is a coupled `correlation`. Sorted order is fine for
    overlap/disjointness checks but NOT for the tuple-position map -- see _coupled_field_symbols."""
    return sorted(
        {
            str(nt)
            for search in req.aggregate.inner_searches.values()
            for nt in search.get_access_points()
        }
    )


def _coupled_field_symbols(req: PopulationRequirement) -> list[str]:
    """The field per tuple *position* of a coupled inner element, e.g. ``[<x>, <y>]`` for
    ``correlation((int(<x>), int(<y>)) for ...)`` -- position order, NOT sorted. Construction sorts
    position-0's values and grafts position-0's field, so the two must line up; using the sorted set
    would mispair a reversed tuple and collapse the correlation."""
    elt = ast.parse(req.aggregate.inner_expression, mode="eval").body
    if not isinstance(elt, ast.Tuple) or len(elt.elts) != 2:
        raise NotImplementedError(
            "`correlation`'s inner element must be a 2-tuple, e.g. "
            "correlation((int(<x>), int(<y>)) for x in population)."
        )
    searches = req.aggregate.inner_searches
    ordered: list[str] = []
    for sub in elt.elts:
        names = {n.id for n in ast.walk(sub) if isinstance(n, ast.Name)} & set(searches)
        access = sorted(
            {str(nt) for name in names for nt in searches[name].get_access_points()}
        )
        if len(access) != 1:
            raise NotImplementedError(
                f"Each slot of `correlation`'s (x, y) tuple must read exactly one field; got "
                f"{access or 'none'}."
            )
        ordered.append(access[0])
    return ordered


def _is_prefix(shorter: tuple, longer: tuple) -> bool:
    """True if ``shorter`` is a (proper or equal) path prefix of ``longer``."""
    return len(shorter) <= len(longer) and tuple(longer[: len(shorter)]) == tuple(shorter)


class PopulationShortfallError(FandangoValueError):
    """Raised when the sampler cannot assemble a batch meeting a requirement within budget."""


class PopulationSampler:
    """Construct a batch of individuals satisfying a grammar's hard population requirements.

    :param grammar: the grammar to fuzz from; its ``population_requirements`` are used by
        default.
    :param requirements: override the requirement list (defaults to
        ``grammar.population_requirements``).
    :param max_attempts_per_slot: fuzzing budget per individual before declaring a shortfall.
    """

    #: Shortfall policies for :meth:`sample`. ``fail_loud`` raises; ``best_effort`` returns the
    #: closest assembled batch (or a plain valid batch) with a structured warning.
    _SHORTFALL_POLICIES = frozenset({"fail_loud", "best_effort"})

    def __init__(
        self,
        grammar: Grammar,
        requirements: Optional[list[PopulationRequirement]] = None,
        *,
        constraints: Optional[list[Constraint]] = None,
        max_attempts_per_slot: int = 1000,
        distribution_pool_factor: int = 40,
        on_shortfall: str = "fail_loud",
        correlation_tolerance: float = 0.15,
    ) -> None:
        self.grammar = grammar
        self.requirements = (
            list(requirements)
            if requirements is not None
            else list(getattr(grammar, "population_requirements", []))
        )
        # Per-tree hard constraints every constructed individual must satisfy (co-enforced by
        # rejection-fuzzing the candidate source below).
        self._constraints = list(constraints) if constraints else []
        self._max_attempts_per_slot = max(1, max_attempts_per_slot)
        # How many candidate individuals to fuzz per requested one when matching a target
        # distribution: a larger pool covers the target quantiles more finely (better fit).
        self._distribution_pool_factor = max(1, distribution_pool_factor)
        if on_shortfall not in self._SHORTFALL_POLICIES:
            raise FandangoValueError(
                f"on_shortfall must be one of {sorted(self._SHORTFALL_POLICIES)}; "
                f"got {on_shortfall!r}. (relax_to_nearest_feasible is future work.)"
            )
        self._on_shortfall = on_shortfall
        # Exact `correlation == r` is essentially never achievable on discrete grammar values, so
        # the gate accepts within this tolerance (the correlation analogue of the fraction `==`
        # snap-and-warn). Inequalities ignore it.
        self._correlation_tolerance = abs(correlation_tolerance)
        # The closest batch assembled during the current sample(), surfaced by best_effort when a
        # gate later fails (the construction targets the nearest feasible, so it is the best miss).
        self._last_batch: Optional[list[DerivationTree]] = None
        # The environment the inner predicate is eval'd in (so spec-level `def`s/imports work).
        self._global_variables, self._local_variables = grammar.get_spec_env()

    def sample(self, n: int) -> list[DerivationTree]:
        """Return a batch of ``n`` individuals satisfying the population requirements.

        On an unmet requirement the behavior follows ``on_shortfall``: ``fail_loud`` (default)
        raises :class:`PopulationShortfallError`; ``best_effort`` logs a structured warning and
        returns the closest assembled batch (or, if construction could not start, a plain valid
        batch of size ``n``) so the caller gets output plus an explicit note that the guarantee
        did not hold."""
        if n <= 0:
            raise FandangoValueError(f"Population size must be positive; got {n}.")
        if not self.requirements:
            return [self._candidate() for _ in range(n)]
        for req in self.requirements:
            self._validate_requirement(req)
        self._last_batch = None
        try:
            # A lone single-field requirement materializes whole source individuals directly; a
            # coupled requirement (or several) needs the graft path, which sets each field into a
            # shared skeleton.
            if len(self.requirements) == 1 and not self._is_coupled(self.requirements[0]):
                return self._sample_single(self.requirements[0], n)
            self._validate_disjoint()
            return self._sample_joint(n)
        except PopulationShortfallError as shortfall:
            if self._on_shortfall != "best_effort":
                raise
            LOGGER.warning(
                f"Population requirement not fully met (on_shortfall=best_effort); returning the "
                f"closest batch. Shortfall: {shortfall}"
            )
            if self._last_batch is not None and len(self._last_batch) == n:
                return self._last_batch
            # Construction could not even start (e.g. empty candidate pool); fall back to a plain
            # valid batch so the caller still gets n individuals satisfying the per-tree rules.
            return [self._candidate() for _ in range(n)]

    @staticmethod
    def _is_coupled(req: PopulationRequirement) -> bool:
        """A requirement that jointly constrains several fields of one individual (only
        ``correlation`` in v1)."""
        handler = _handler_for(req.aggregate.reducer_name)
        return handler is not None and handler.coupled

    def _validate_requirement(self, req: PopulationRequirement) -> None:
        """Reject, up front, requirement shapes v1 cannot construct: unsupported operators or
        reducers, an ``>=``/``>``/``==`` on a distributional fit (a distance has no such target),
        and multi-symbol (row-scoped/coupled) inner elements. Per-reducer rules come from the
        handler registry (see ``_BUILTIN_HANDLERS``)."""
        if req.operator not in _CONSTRUCTIBLE_OPERATORS:
            raise NotImplementedError(
                f"Operator '{req.operator.value}' is not supported for a population requirement."
            )
        reducer = req.aggregate.reducer_name
        handler = _handler_for(reducer)
        if handler is None:
            raise NotImplementedError(
                f"v1 constructs `fraction` quotas, `distinct_count` diversity, distributional "
                f"fits ({', '.join(sorted(_DISTRIBUTION_QUANTILES))}), and `correlation`; got "
                f"'{reducer}'."
            )
        if req.operator not in handler.allowed_operators:
            raise NotImplementedError(
                f"Operator '{req.operator.value}' is not supported for '{reducer}': "
                f"{handler.operator_hint}."
            )
        grouping = grouping_for(reducer)
        if grouping != "pool":
            raise NotImplementedError(
                f"'{reducer}' declares grouping='{grouping}', but the sampler constructs only "
                f"pooled (single-value-per-individual) requirements in v1; constructing a "
                f"'{grouping}' batch is future work. (The soft objective path honors it.)"
            )
        symbols = _requirement_symbols(req)
        if len(symbols) != handler.arity:
            if handler.coupled:
                raise NotImplementedError(
                    f"`{reducer}` constrains exactly {handler.arity} fields (its (x, y) tuple); "
                    f"'{reducer}(...)' reads {len(symbols)} ({', '.join(symbols)})."
                )
            raise NotImplementedError(
                f"v1 constructs single-field requirements; '{reducer}(...)' reads "
                f"{len(symbols)} symbols ({', '.join(symbols)}). Row-scoped/coupled requirements "
                f"beyond `correlation` are future work."
            )

    def _validate_disjoint(self) -> None:
        """Requirements must target distinct fields (any shared symbol = same-field joint, out of
        scope -- e.g. a coupled field also used by a marginal requirement). Nested-field
        (containment) collisions are caught structurally at graft time."""
        seen: dict[str, str] = {}
        for req in self.requirements:
            for symbol in _requirement_symbols(req):
                if symbol in seen:
                    raise FandangoValueError(
                        f"Two population requirements target the same field {symbol} "
                        f"('{seen[symbol]}' and '{req.aggregate.reducer_name}'). Jointly "
                        f"constraining one field with several requirements is future work; use "
                        f"disjoint fields."
                    )
                seen[symbol] = req.aggregate.reducer_name

    # -- dispatch: a per-slot {symbol: source individual} column ------------- #
    def _plan(
        self, req: PopulationRequirement, n: int
    ) -> list[dict[str, DerivationTree]]:
        """A length-``n`` column of per-slot ``{symbol: carrier}`` maps: grafting each carrier's
        ``symbol`` field into ``n`` individuals makes the requirement hold. Single-field reducers
        yield one-key maps (carrier = a whole valid individual); coupled ``correlation`` yields
        two-key maps."""
        handler = _handler_for(req.aggregate.reducer_name)
        assert handler is not None  # guaranteed by _validate_requirement
        if handler.coupled:
            return handler.construct(self, req, n)  # already a column of {symbol: carrier} maps
        column = handler.construct(self, req, n)  # list of whole carriers
        symbol = _requirement_symbols(req)[0]
        return [{symbol: tree} for tree in column]

    def _sample_single(self, req: PopulationRequirement, n: int) -> list[DerivationTree]:
        """One single-field requirement: each source is already a whole valid individual, so
        materialize the column directly (deep-copy) and verify -- no grafting needed."""
        batch = [
            next(iter(slot.values())).deepcopy(copy_parent=False)
            for slot in self._plan(req, n)
        ]
        self._last_batch = batch  # closest miss for best_effort, before the gate can raise
        self._verify_dispatch(req, batch)
        return batch

    def _verify_dispatch(
        self, req: PopulationRequirement, batch: list[DerivationTree]
    ) -> None:
        handler = _handler_for(req.aggregate.reducer_name)
        assert handler is not None  # guaranteed by _validate_requirement
        handler.verify(self, req, self._inner(req), batch)

    # -- graft construction: set each requirement's field(s) into skeletons -- #
    def _sample_joint(self, n: int) -> list[DerivationTree]:
        """Graft-based construction (several disjoint-field requirements, or a lone coupled one):
        plan a per-slot ``{symbol: carrier}`` column per requirement, then graft every requirement's
        field(s) into ``n`` shared skeleton individuals. Disjoint fields don't interfere, so each
        requirement's verify gate holds independently; a coupled requirement's two fields are grafted
        together so the pair holds per individual."""
        skeletons = [self._candidate() for _ in range(n)]
        columns = {i: self._plan(req, n) for i, req in enumerate(self.requirements)}
        batch = self._graft_all(skeletons, columns)
        self._last_batch = batch  # closest miss for best_effort, before any gate can raise
        for req in self.requirements:
            self._verify_dispatch(req, batch)
        self._recheck_constraints(batch)
        return batch

    def _graft_all(
        self,
        skeletons: list[DerivationTree],
        columns: dict[int, list[dict[str, DerivationTree]]],
    ) -> list[DerivationTree]:
        batch: list[DerivationTree] = []
        for slot, skeleton in enumerate(skeletons):
            # Every (symbol -> carrier) graft for this individual, across all requirements (a
            # coupled requirement contributes two).
            grafts = [
                (symbol, carrier)
                for i in range(len(self.requirements))
                for symbol, carrier in columns[i][slot].items()
            ]
            symbols = [symbol for symbol, _ in grafts]
            nodes = []
            for symbol in symbols:
                found = list(skeleton.find_subtrees(symbol))
                if len(found) != 1:
                    raise NotImplementedError(
                        f"v1 requires exactly one {symbol} per individual for grafting; got "
                        f"{len(found)}. Multi-valued fields need the grouping policy (future work)."
                    )
                nodes.append(found[0])
            self._reject_nested(nodes, symbols)
            replacements = []
            for (symbol, carrier), old in zip(grafts, nodes):
                if old.read_only:
                    raise NotImplementedError(
                        f"The population field {symbol} is under a generator output (read-only) "
                        f"and cannot be grafted; this is future work."
                    )
                source_field = next(carrier.find_subtrees(symbol))
                replacements.append((old, source_field.deepcopy(copy_parent=False)))
            batch.append(skeleton.replace_multiple(self.grammar, replacements))
        return batch

    def _reject_nested(
        self, nodes: list[DerivationTree], symbols: list[str]
    ) -> None:
        """Distinct symbols can still collide if one field derives *under* another; a graft into
        the outer node would clobber the inner. Detect it structurally: reject if any node's choice
        path is a prefix of another's."""
        paths = [node.get_choices_path() for node in nodes]
        for a in range(len(paths)):
            for b in range(len(paths)):
                if a != b and _is_prefix(paths[a], paths[b]):
                    raise FandangoValueError(
                        f"Population requirement fields {symbols[a]} and {symbols[b]} are nested "
                        f"(one derives under the other); v1 needs structurally disjoint fields."
                    )

    def _recheck_constraints(self, batch: list[DerivationTree]) -> None:
        """A grafted individual mixes the skeleton's untouched fields with a grafted field; if a
        per-tree constraint reads a grafted field it may now be violated. Re-check every tree and
        shortfall rather than silently emit an invalid individual."""
        if not self._constraints:
            return
        for tree in batch:
            if not all(constraint.check(tree) for constraint in self._constraints):
                raise PopulationShortfallError(
                    "A grafted individual violates a per-tree constraint (a constraint reads a "
                    "field set by a population requirement). Per-slot validity-aware selection is "
                    "future work; make the per-tree constraint and the requirement target "
                    "different fields."
                )

    def _candidate(self) -> DerivationTree:
        """A freshly fuzzed individual that satisfies every per-tree hard constraint.

        With no constraints this is a plain ``grammar.fuzz()``. Otherwise it rejection-fuzzes:
        the population sampler bypasses the GA, so per-tree constraints are enforced by drawing
        until a valid individual appears. A too-tight constraint exhausts the budget -> shortfall."""
        if not self._constraints:
            return self.grammar.fuzz()
        for _ in range(self._max_attempts_per_slot):
            tree = self.grammar.fuzz()
            if all(constraint.check(tree) for constraint in self._constraints):
                return tree
        raise PopulationShortfallError(
            f"Could not fuzz an individual satisfying the {len(self._constraints)} per-tree "
            f"constraint(s) within {self._max_attempts_per_slot} attempts. The constraints may "
            f"be too tight for plain fuzzing (GA-backed candidate generation is future work)."
        )

    def _inner(self, req: PopulationRequirement) -> _InnerValue:
        """A per-tree evaluator for the requirement's inner element."""
        return _InnerValue(
            req.aggregate.inner_expression,
            searches=req.aggregate.inner_searches,
            global_variables=self._global_variables,
            local_variables=self._local_variables,
        )

    def _sole_value(self, inner: _InnerValue, tree: DerivationTree) -> Any:
        """The single inner value for one individual (each individual = one record in v1).

        A tree yielding several values is the pooled/multiplicity case (grouping policy) and is
        not yet supported."""
        values = inner.raw_values(tree)
        if len(values) != 1:
            raise NotImplementedError(
                f"v1 requires exactly one inner value per individual, but one tree yielded "
                f"{len(values)}. Multi-valued fields need the grouping policy (future work)."
            )
        return values[0]

    # -- fraction quota: bucket source individuals by the predicate --------- #
    def _fraction_plan(self, req: PopulationRequirement, n: int) -> list[DerivationTree]:
        target_true = self._target_count(req, n)
        target_false = n - target_true
        inner = self._inner(req)

        true_bucket: list[DerivationTree] = []
        false_bucket: list[DerivationTree] = []
        cap = self._max_attempts_per_slot * n
        attempts = 0
        while len(true_bucket) < target_true or len(false_bucket) < target_false:
            if attempts >= cap:
                raise PopulationShortfallError(
                    f"Could not assemble a batch for 'fraction {req.operator.value} "
                    f"{req.bound}' at N={n} within {cap} fuzzing attempts "
                    f"(satisfying: {len(true_bucket)}/{target_true}, "
                    f"violating: {len(false_bucket)}/{target_false}). The predicate may be "
                    f"too rare or too common under plain fuzzing."
                )
            attempts += 1
            tree = self._candidate()
            if self._predicate_value(inner, tree):
                if len(true_bucket) < target_true:
                    true_bucket.append(tree)
            elif len(false_bucket) < target_false:
                false_bucket.append(tree)

        column = true_bucket + false_bucket
        random.shuffle(column)
        return column

    def _target_count(self, req: PopulationRequirement, n: int) -> int:
        """The number of individuals that must satisfy the predicate for ``req`` to hold at N.

        For ``==`` the exact fraction is snapped to the nearest achievable ``round(p*N)`` (R3),
        warning when ``p*N`` is not integral; the inequalities take the boundary count.
        """
        p = req.bound
        if not (0.0 <= p <= 1.0):
            raise FandangoValueError(
                f"A `fraction` target must be in [0, 1]; got {p}."
            )
        exact = p * n
        op = req.operator
        if op is Comparison.EQUAL:
            target = round(exact)
            if target != exact:
                LOGGER.warning(
                    f"Exact fraction {p} is not achievable at N={n}; using {target}/{n} "
                    f"= {target / n:.4f}."
                )
        elif op is Comparison.GREATER_EQUAL:
            target = math.ceil(exact)
        elif op is Comparison.LESS_EQUAL:
            target = math.floor(exact)
        elif op is Comparison.GREATER:
            target = math.floor(exact) + 1
        elif op is Comparison.LESS:
            target = math.ceil(exact) - 1
        else:  # pragma: no cover - guarded by _QUOTA_OPERATORS
            raise NotImplementedError(
                f"Operator '{op.value}' is not supported for a fraction quota."
            )

        if not (0 <= target <= n):
            raise FandangoValueError(
                f"Requirement 'fraction {op.value} {p}' is infeasible at N={n}: it would need "
                f"{target} of {n} individuals to satisfy the predicate."
            )
        return target

    def _predicate_value(self, inner: _InnerValue, tree: DerivationTree) -> bool:
        """Evaluate the inner boolean predicate against one individual."""
        return bool(self._sole_value(inner, tree))

    def _verify(
        self,
        req: PopulationRequirement,
        inner: _InnerValue,
        batch: list[DerivationTree],
        target_true: int,
    ) -> None:
        """Independent gate: the assembled batch must contain exactly the constructed count of
        predicate-satisfying individuals. This is the real invariant -- for ``==`` it means the
        snapped target, for the inequalities the boundary count -- so re-checking it catches any
        bucketing bug regardless of the operator."""
        achieved = sum(1 for tree in batch if self._predicate_value(inner, tree))
        if achieved != target_true:
            raise PopulationShortfallError(
                f"Verification gate failed for 'fraction {req.operator.value} {req.bound}': "
                f"assembled batch has {achieved} satisfying individuals, expected {target_true}."
            )

    # -- distinct-value diversity: gather representatives, fill by reuse ----- #
    def _distinct_plan(self, req: PopulationRequirement, n: int) -> list[DerivationTree]:
        target, mode = self._target_distinct(req, n)
        inner = self._inner(req)

        # `>= 0` (or `< 1` -> handled as infeasible) imposes nothing; a plain batch suffices.
        if target <= 0:
            return [self._candidate() for _ in range(n)]

        # We never need more than N distinct values in an N-tree batch.
        gather_target = min(target, n)
        cap = self._max_attempts_per_slot * n

        # Gather one representative individual per distinct field value. "reach" mode must find
        # `gather_target` distinct values (else the grammar can't meet the requirement ->
        # shortfall); "cap" mode takes up to that many and stops early if fuzzing runs dry.
        reps: dict[Any, DerivationTree] = {}
        attempts = 0
        while len(reps) < gather_target:
            if attempts >= cap:
                if mode == "reach":
                    raise PopulationShortfallError(
                        f"Could not find {target} distinct values for "
                        f"'distinct_count {req.operator.value} {req.bound}' at N={n} within {cap} "
                        f"fuzzing attempts (found {len(reps)}). The grammar may not produce that "
                        f"many distinct values."
                    )
                break  # "cap": fewer distinct values than the cap is fine (<=/< still holds)
            attempts += 1
            tree = self._candidate()
            value = self._sole_value(inner, tree)
            if value not in reps:
                reps[value] = tree

        rep_trees = list(reps.values())
        column: list[DerivationTree] = list(rep_trees)
        # Fill the remaining slots by *reusing* representatives (same source object repeated -- the
        # materialization/graft step deep-copies, so entries stay independent). This keeps the
        # distinct count exactly len(reps) without re-fuzzing, so a high-cardinality grammar can't
        # cause a false shortfall.
        i = 0
        while len(column) < n:
            column.append(rep_trees[i % len(rep_trees)])
            i += 1

        random.shuffle(column)
        return column

    def _target_distinct(self, req: PopulationRequirement, n: int) -> tuple[int, str]:
        """The distinct-value count to construct toward, and the mode:

        - ``"reach"`` (``>=``/``>``/``==``): build *exactly* this many distinct values (must be
          found, or it is a shortfall). Satisfies the operator since e.g. ``== ceil`` for ``>=``.
        - ``"cap"`` (``<=``/``<``): use *at most* this many distinct values (fewer is fine).
        """
        b = req.bound
        if b < 0:
            raise FandangoValueError(
                f"A `distinct_count` target must be non-negative; got {b}."
            )
        op = req.operator
        if op is Comparison.GREATER_EQUAL:
            target, mode = math.ceil(b), "reach"
        elif op is Comparison.GREATER:
            target, mode = math.floor(b) + 1, "reach"
        elif op is Comparison.EQUAL:
            if b != int(b):
                raise FandangoValueError(
                    f"'distinct_count == {b}' can never hold: a count is an integer."
                )
            target, mode = int(b), "reach"
        elif op is Comparison.LESS_EQUAL:
            target, mode = math.floor(b), "cap"
        elif op is Comparison.LESS:
            target, mode = math.ceil(b) - 1, "cap"
        else:  # pragma: no cover - guarded by _CONSTRUCTIBLE_OPERATORS
            raise NotImplementedError(
                f"Operator '{op.value}' is not supported for a distinct_count requirement."
            )

        if mode == "reach" and target > n:
            raise FandangoValueError(
                f"Requirement 'distinct_count {op.value} {b}' is infeasible at N={n}: a batch of "
                f"{n} individuals can hold at most {n} distinct values."
            )
        if mode == "cap" and target < 1:
            raise FandangoValueError(
                f"Requirement 'distinct_count {op.value} {b}' is infeasible: a non-empty batch "
                f"always has at least one distinct value."
            )
        return target, mode

    def _verify_distinct(
        self, req: PopulationRequirement, inner: _InnerValue, batch: list[DerivationTree]
    ) -> None:
        """Independent gate: the assembled batch's distinct-value count must satisfy the
        operator. ``distinct_count`` targets are integers (no snapping), so the literal operator
        check is exact."""
        values = [self._sole_value(inner, tree) for tree in batch]
        actual = REDUCERS["distinct_count"](values)
        if not req.operator.compare(actual, req.bound):
            raise PopulationShortfallError(
                f"Verification gate failed for 'distinct_count {req.operator.value} {req.bound}': "
                f"assembled batch has {actual} distinct values."
            )

    # -- distributional shape: select the pool tree nearest each quantile --- #
    def _select_by_targets(
        self,
        req: PopulationRequirement,
        targets: list[float],
    ) -> list[DerivationTree]:
        """Pin a column to a list of ``targets``: fuzz a candidate pool, then for each target
        value pick the individual whose (sole) field value is nearest it. Shared by the built-in
        distributional fit (targets = the target quantiles) and custom ``sample`` handlers
        (targets = the handler's drawn values). The column's order statistics land on the targets
        as closely as the grammar's own values allow, without assuming how the field is spelled.
        """
        inner = self._inner(req)
        n = len(targets)

        # Candidate pool of (value, tree). Skip trees that don't yield exactly one numeric value.
        pool: list[tuple[float, DerivationTree]] = []
        target_pool = self._distribution_pool_factor * n
        cap = self._max_attempts_per_slot * n
        attempts = 0
        while len(pool) < target_pool and attempts < cap:
            attempts += 1
            tree = self._candidate()
            value = self._sole_value(inner, tree)
            pool.append((float(value), tree))
        if not pool:
            raise PopulationShortfallError(
                f"Could not build a candidate pool for '{req.aggregate.reducer_name} "
                f"{req.operator.value} {req.bound}' at N={n}."
            )

        pool.sort(key=lambda vt: vt[0])
        pool_values = [v for v, _ in pool]
        column: list[DerivationTree] = []
        for t in targets:
            j = bisect.bisect_left(pool_values, t)
            nearest = min(
                (k for k in (j - 1, j, j + 1) if 0 <= k < len(pool)),
                key=lambda k: abs(pool_values[k] - t),
            )
            column.append(pool[nearest][1])

        random.shuffle(column)
        return column

    def _distribution_plan(
        self, req: PopulationRequirement, n: int
    ) -> list[DerivationTree]:
        """Match a target distribution: ``normal_fit([<x> for x in population], ...) <= delta``.

        The fit is the mean gap between the sorted batch and the target quantiles Q((k+0.5)/N),
        so we select the pool candidate nearest each quantile (see :meth:`_select_by_targets`).
        The batch's order statistics land on the target quantiles as closely as the grammar's own
        values allow -- driving the fit to its discretization floor. If the grammar is too coarse
        to reach ``delta``, the verification gate reports a shortfall rather than a false success.
        """
        quantile = _DISTRIBUTION_QUANTILES[req.aggregate.reducer_name](
            *req.aggregate.reducer_args
        )
        return self._select_by_targets(req, [quantile((i + 0.5) / n) for i in range(n)])

    # -- custom registered handler: pin a column to the handler's drawn values -- #
    def _sample_value_plan(
        self, req: PopulationRequirement, n: int
    ) -> list[DerivationTree]:
        """Construct toward a custom ``register_requirement`` handler: draw ``n`` target values
        from the handler's ``sample`` and pin a column nearest them (a marginal construction)."""
        handler = REQUIREMENT_HANDLERS[req.aggregate.reducer_name]
        assert handler.sample is not None  # only sample-bearing handlers reach here
        targets = handler.sample(n, *req.aggregate.reducer_args)
        if len(targets) != n:
            raise PopulationShortfallError(
                f"Custom sampler for '{req.aggregate.reducer_name}' returned {len(targets)} "
                f"values, expected {n}."
            )
        return self._select_by_targets(req, [float(t) for t in targets])

    def _verify_custom(
        self, req: PopulationRequirement, inner: _InnerValue, batch: list[DerivationTree]
    ) -> None:
        """Verify a custom requirement: the handler's ``check`` aggregate must satisfy the
        operator, with the handler's optional ``floor`` giving a precise "unsatisfiable in
        principle" diagnosis (mirroring the built-in fit gate)."""
        handler = REQUIREMENT_HANDLERS[req.aggregate.reducer_name]
        values = [self._sole_value(inner, tree) for tree in batch]
        params = req.aggregate.reducer_args
        actual = handler.check(values, *params)
        if handler.floor is not None:
            floor = float(handler.floor(values, *params))
            if float(req.bound) < floor:
                raise PopulationShortfallError(
                    f"Requirement '{req.aggregate.reducer_name} {req.operator.value} {req.bound}' "
                    f"is unsatisfiable in principle: bound {req.bound} is below the handler's "
                    f"floor ~{floor:.4f} for this field."
                )
        if not req.operator.compare(actual, req.bound):
            raise PopulationShortfallError(
                f"Verification gate failed for '{req.aggregate.reducer_name} "
                f"{req.operator.value} {req.bound}': assembled batch has {actual:.4f}."
            )

    def _verify_fit(
        self, req: PopulationRequirement, inner: _InnerValue, batch: list[DerivationTree]
    ) -> None:
        """Independent gate: the assembled batch's distributional fit must satisfy the operator.

        On this by-construction path the batch's order statistics are *placed* on the target
        quantiles, so the fit distance IS the achieved point estimate — that is the right gate
        here (see the module note on why a bootstrap CI, which assumes an iid draw, is not:
        resampling a deliberately quantile-placed batch wildly overstates the distance). Two
        distinct failure modes are separated:

        * ``delta`` below the **discretization floor** ``~h/4`` (``h`` = the field's value step):
          unsatisfiable *in principle* — no placement of a grid-valued field can get that close
          to a continuous target. Reported precisely, not as a generic "coarse grammar".
        * otherwise the grammar is simply too coarse to reach ``delta`` at this N -- a shortfall.
        """
        values = [float(self._sole_value(inner, tree)) for tree in batch]
        actual = REDUCERS[req.aggregate.reducer_name](values, *req.aggregate.reducer_args)
        below_floor, floor = is_delta_below_floor(float(req.bound), values)
        if below_floor:
            raise PopulationShortfallError(
                f"Requirement '{req.aggregate.reducer_name} {req.operator.value} {req.bound}' "
                f"is unsatisfiable in principle: delta={req.bound} is below the discretization "
                f"floor ~{floor:.4f} for this field (its values step by ~{4 * floor:.4g}, so a "
                f"rounded field sits at least ~{floor:.4f} from a continuous target no matter "
                f"how it is placed). Raise delta to at least ~{floor:.4f}."
            )
        if not req.operator.compare(actual, req.bound):
            raise PopulationShortfallError(
                f"Verification gate failed for '{req.aggregate.reducer_name} "
                f"{req.operator.value} {req.bound}': assembled batch has fit {actual:.4f} "
                f"(discretization floor ~{floor:.4f}). The grammar may be too coarse to match "
                f"the target distribution that closely at N={len(batch)}."
            )

    # -- coupled correlation: pair two fields to reach the target correlation - #
    def _correlation_plan(
        self, req: PopulationRequirement, n: int
    ) -> list[dict[str, DerivationTree]]:
        """Construct N (x, y) pairs whose Pearson correlation meets the bound, by pairing the two
        fields' fuzzed values monotonically (for ``>=``/``>``, driving toward +1) or anti-monotonically
        (for ``<=``/``<``, toward -1). Each slot maps the position-0 field to the source whose x is at
        that rank and the position-1 field to the source whose y is at the matching rank; grafting
        both sets the pair per individual. If the marginals are too coarse (low variance/ties) to
        reach the bound, the verify gate reports a shortfall.
        """
        sym_x, sym_y = _coupled_field_symbols(req)  # tuple-position order (NOT sorted)
        inner = self._inner(req)

        # Pool = N. Oversampling doesn't help: reachability is bounded by marginal variance/ties,
        # which more candidates can't change (unlike the quantile-coverage distribution pool).
        pool: list[tuple[float, float, DerivationTree]] = []
        for _ in range(n):
            tree = self._candidate()
            x, y = self._sole_value(inner, tree)  # one (x, y) pair, tuple order
            pool.append((float(x), float(y), tree))

        xs = sorted(pool, key=lambda z: z[0])  # (x, y, tree) by x ascending
        ys = sorted(pool, key=lambda z: z[1])  # by y ascending
        x_vals, x_trees = [p[0] for p in xs], [p[2] for p in xs]
        y_vals, y_trees = [p[1] for p in ys], [p[2] for p in ys]

        # `== r` targets a *specific* correlation; the inequalities just drive to the extreme
        # (monotone for >=/>, anti-monotone for <=/<) that satisfies them.
        if req.operator is Comparison.EQUAL:
            rank_x, rank_y = self._search_copula_ranks(
                n, float(req.bound), x_vals, y_vals
            )
        else:
            ascending = req.operator in (Comparison.GREATER, Comparison.GREATER_EQUAL)
            rank_x = list(range(n))
            rank_y = list(range(n)) if ascending else list(range(n - 1, -1, -1))

        slots = [
            {sym_x: x_trees[rank_x[i]], sym_y: y_trees[rank_y[i]]} for i in range(n)
        ]
        random.shuffle(slots)  # shuffle whole maps -- pairing is preserved within each
        return slots

    def _search_copula_ranks(
        self, n: int, r: float, x_vals: list[float], y_vals: list[float]
    ) -> tuple[list[int], list[int]]:
        """Pick the copula rank pairing whose *achieved* correlation is closest to ``r``.

        A single Gaussian-copula draw approximates ``r`` but with ~0.1 sampling noise on discrete
        marginals, which is too coarse for ``== r``. Drawing many candidate rankings and keeping
        the one whose actual (x, y) Pearson correlation is nearest ``r`` tightens this to the
        grammar's achievable resolution -- cheap, since scoring a ranking is O(n) and needs no new
        fuzzing. Stops early once a draw lands within the tolerance."""
        attempts = max(1, self._distribution_pool_factor) * 5  # e.g. 200 draws by default
        best_ranks: Optional[tuple[list[int], list[int]]] = None
        best_error = float("inf")
        for _ in range(attempts):
            rx, ry = self._copula_ranks(n, r)
            achieved = REDUCERS["correlation"](
                [(x_vals[rx[i]], y_vals[ry[i]]) for i in range(n)]
            )
            error = abs(achieved - r)
            if error < best_error:
                best_error, best_ranks = error, (rx, ry)
                if error <= self._correlation_tolerance:
                    break
        assert best_ranks is not None
        return best_ranks

    @staticmethod
    def _copula_ranks(n: int, r: float) -> tuple[list[int], list[int]]:
        """Rank assignments that pair x-rank with y-rank at approximately correlation ``r``.

        Draw ``n`` latent pairs from a bivariate normal with correlation ``r`` (``ly = r*z1 +
        sqrt(1-r^2)*z2``); slot ``i`` then takes the x-value at ``rank_x[i]`` and the y-value at
        ``rank_y[i]``, where the ranks are the latent pair's ranks. Pairing the sorted marginals
        by these correlated ranks reproduces ``r`` in the batch (a Gaussian copula), which is what
        lets construction hit an arbitrary target rather than only the +/-1 extremes.
        """
        r = max(-0.999, min(0.999, r))
        c = math.sqrt(1.0 - r * r)
        latents = []
        for i in range(n):
            z1 = random.gauss(0.0, 1.0)
            z2 = random.gauss(0.0, 1.0)
            latents.append((z1, r * z1 + c * z2, i))
        rank_x = [0] * n
        rank_y = [0] * n
        for rank, (_, _, idx) in enumerate(sorted(latents, key=lambda l: l[0])):
            rank_x[idx] = rank
        for rank, (_, _, idx) in enumerate(sorted(latents, key=lambda l: l[1])):
            rank_y[idx] = rank
        return rank_x, rank_y

    def _verify_correlation(
        self, req: PopulationRequirement, inner: _InnerValue, batch: list[DerivationTree]
    ) -> None:
        """Independent gate: the assembled batch's (x, y) correlation must satisfy the operator.
        ``_correlation`` returns 0.0 for a constant column, so an unreachable bound (e.g. a field
        with no variance) surfaces here as a shortfall rather than a false success.

        For ``== r`` the gate is a tolerance band (``|actual - r| <= correlation_tolerance``):
        exact Pearson equality is essentially never achievable on discrete values, so this is the
        correlation analogue of the fraction ``==`` snap-and-warn."""
        pairs = [self._sole_value(inner, tree) for tree in batch]
        actual = REDUCERS["correlation"](pairs)
        if req.operator is Comparison.EQUAL:
            if abs(actual - float(req.bound)) > self._correlation_tolerance:
                raise PopulationShortfallError(
                    f"Verification gate failed for 'correlation == {req.bound}': assembled batch "
                    f"has correlation {actual:.4f}, outside the +/-{self._correlation_tolerance} "
                    f"tolerance. The marginals may be too coarse (low variance or ties) to reach "
                    f"that correlation."
                )
            return
        if not req.operator.compare(actual, req.bound):
            raise PopulationShortfallError(
                f"Verification gate failed for 'correlation {req.operator.value} {req.bound}': "
                f"assembled batch has correlation {actual:.4f}. The marginals may be too coarse "
                f"(low variance or ties) to reach that bound."
            )
