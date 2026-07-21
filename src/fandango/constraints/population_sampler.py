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

Coupled `correlation((<x>, <y>) for x in population) OP r` (`>=`/`>`/`<=`/`<`) is constructed by
monotone (or anti-monotone) pairing: fuzz a pool, sort each field's values, pair the i-th of each,
and graft both fields together into a skeleton so the pair holds per individual. Exact `== r`,
conditional `P(y|x)`, and >2-way coupling are future work.
"""

import ast
import bisect
import math
import random
from statistics import NormalDist
from typing import Any, Optional

from fandango.constraints.constraint import Constraint
from fandango.constraints.failing_tree import Comparison
from fandango.constraints.population import (
    REDUCERS,
    PopulationRequirement,
    _InnerValue,
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

    def __init__(
        self,
        grammar: Grammar,
        requirements: Optional[list[PopulationRequirement]] = None,
        *,
        constraints: Optional[list[Constraint]] = None,
        max_attempts_per_slot: int = 1000,
        distribution_pool_factor: int = 40,
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
        # The environment the inner predicate is eval'd in (so spec-level `def`s/imports work).
        self._global_variables, self._local_variables = grammar.get_spec_env()

    def sample(self, n: int) -> list[DerivationTree]:
        """Return a batch of ``n`` individuals satisfying the population requirements."""
        if n <= 0:
            raise FandangoValueError(f"Population size must be positive; got {n}.")
        if not self.requirements:
            return [self._candidate() for _ in range(n)]
        for req in self.requirements:
            self._validate_requirement(req)
        # A lone single-field requirement materializes whole source individuals directly; a coupled
        # requirement (or several requirements) needs the graft path, which sets each field into a
        # shared skeleton.
        if len(self.requirements) == 1 and not self._is_coupled(self.requirements[0]):
            return self._sample_single(self.requirements[0], n)
        self._validate_disjoint()
        return self._sample_joint(n)

    @staticmethod
    def _is_coupled(req: PopulationRequirement) -> bool:
        """A requirement that jointly constrains several fields of one individual (only
        ``correlation`` in v1)."""
        return req.aggregate.reducer_name == "correlation"

    def _validate_requirement(self, req: PopulationRequirement) -> None:
        """Reject, up front, requirement shapes v1 cannot construct: unsupported operators or
        reducers, an ``>=``/``>``/``==`` on a distributional fit (a distance has no such target),
        and multi-symbol (row-scoped/coupled) inner elements."""
        if req.operator not in _CONSTRUCTIBLE_OPERATORS:
            raise NotImplementedError(
                f"Operator '{req.operator.value}' is not supported for a population requirement."
            )
        reducer = req.aggregate.reducer_name
        constructible = {"fraction", "distinct_count", "correlation"} | set(
            _DISTRIBUTION_QUANTILES
        )
        if reducer not in constructible:
            raise NotImplementedError(
                f"v1 constructs `fraction` quotas, `distinct_count` diversity, distributional "
                f"fits ({', '.join(sorted(_DISTRIBUTION_QUANTILES))}), and `correlation`; got "
                f"'{reducer}'."
            )
        if reducer in _DISTRIBUTION_QUANTILES and req.operator not in (
            Comparison.LESS_EQUAL,
            Comparison.LESS,
        ):
            raise NotImplementedError(
                f"A distributional fit is a distance to a target; only `<=`/`<` are meaningful "
                f"(match within delta). Got '{req.operator.value}' for '{reducer}'."
            )
        if reducer == "correlation" and req.operator is Comparison.EQUAL:
            raise NotImplementedError(
                "Constructing toward an exact correlation is future work; use `>=`/`>`/`<=`/`<`."
            )
        symbols = _requirement_symbols(req)
        if reducer == "correlation":
            if len(symbols) != 2:
                raise NotImplementedError(
                    f"`correlation` constrains exactly two fields (its (x, y) tuple); "
                    f"'{reducer}(...)' reads {len(symbols)} ({', '.join(symbols)})."
                )
        elif len(symbols) != 1:
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
        reducer = req.aggregate.reducer_name
        if reducer == "correlation":
            return self._correlation_plan(req, n)
        if reducer == "fraction":
            column = self._fraction_plan(req, n)
        elif reducer == "distinct_count":
            column = self._distinct_plan(req, n)
        else:
            column = self._distribution_plan(req, n)  # validated to be a distributional fit
        symbol = _requirement_symbols(req)[0]
        return [{symbol: tree} for tree in column]

    def _sample_single(self, req: PopulationRequirement, n: int) -> list[DerivationTree]:
        """One single-field requirement: each source is already a whole valid individual, so
        materialize the column directly (deep-copy) and verify -- no grafting needed."""
        batch = [
            next(iter(slot.values())).deepcopy(copy_parent=False)
            for slot in self._plan(req, n)
        ]
        self._verify_dispatch(req, batch)
        return batch

    def _verify_dispatch(
        self, req: PopulationRequirement, batch: list[DerivationTree]
    ) -> None:
        inner = self._inner(req)
        reducer = req.aggregate.reducer_name
        if reducer == "fraction":
            self._verify(req, inner, batch, self._target_count(req, len(batch)))
        elif reducer == "distinct_count":
            self._verify_distinct(req, inner, batch)
        elif reducer == "correlation":
            self._verify_correlation(req, inner, batch)
        else:
            self._verify_fit(req, inner, batch)

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
    def _distribution_plan(
        self, req: PopulationRequirement, n: int
    ) -> list[DerivationTree]:
        """Match a target distribution: ``normal_fit([<x> for x in population], ...) <= delta``.

        The fit is the mean gap between the sorted batch and the target quantiles Q((k+0.5)/N),
        so we fuzz a candidate pool, then for each quantile pick the candidate whose value is
        nearest to it. The batch's order statistics land on the target quantiles as closely as the
        grammar's own values allow -- driving the fit to its discretization floor -- without
        assuming anything about how the field is spelled. If the grammar is too coarse to reach
        ``delta``, the verification gate reports a shortfall rather than a false success.
        """
        reducer = req.aggregate.reducer_name
        quantile = _DISTRIBUTION_QUANTILES[reducer](*req.aggregate.reducer_args)
        inner = self._inner(req)

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
                f"Could not build a candidate pool for '{reducer} {req.operator.value} "
                f"{req.bound}' at N={n}."
            )

        pool.sort(key=lambda vt: vt[0])
        pool_values = [v for v, _ in pool]
        column: list[DerivationTree] = []
        for i in range(n):
            q = quantile((i + 0.5) / n)
            j = bisect.bisect_left(pool_values, q)
            nearest = min(
                (k for k in (j - 1, j, j + 1) if 0 <= k < len(pool)),
                key=lambda k: abs(pool_values[k] - q),
            )
            column.append(pool[nearest][1])

        random.shuffle(column)
        return column

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
        ascending = req.operator in (Comparison.GREATER, Comparison.GREATER_EQUAL)

        # Pool = N. Oversampling doesn't help: reachability is bounded by marginal variance/ties,
        # which more candidates can't change (unlike the quantile-coverage distribution pool).
        pool: list[tuple[float, float, DerivationTree]] = []
        for _ in range(n):
            tree = self._candidate()
            x, y = self._sole_value(inner, tree)  # one (x, y) pair, tuple order
            pool.append((float(x), float(y), tree))

        x_cands = sorted(((x, t) for x, _, t in pool), key=lambda z: z[0])
        y_cands = sorted(
            ((y, t) for _, y, t in pool), key=lambda z: z[0], reverse=not ascending
        )
        slots = [
            {sym_x: x_cands[i][1], sym_y: y_cands[i][1]} for i in range(n)
        ]
        random.shuffle(slots)  # shuffle whole maps -- pairing is preserved within each
        return slots

    def _verify_correlation(
        self, req: PopulationRequirement, inner: _InnerValue, batch: list[DerivationTree]
    ) -> None:
        """Independent gate: the assembled batch's (x, y) correlation must satisfy the operator.
        ``_correlation`` returns 0.0 for a constant column, so an unreachable bound (e.g. a field
        with no variance) surfaces here as a shortfall rather than a false success."""
        pairs = [self._sole_value(inner, tree) for tree in batch]
        actual = REDUCERS["correlation"](pairs)
        if not req.operator.compare(actual, req.bound):
            raise PopulationShortfallError(
                f"Verification gate failed for 'correlation {req.operator.value} {req.bound}': "
                f"assembled batch has correlation {actual:.4f}. The marginals may be too coarse "
                f"(low variance or ties) to reach that bound."
            )
