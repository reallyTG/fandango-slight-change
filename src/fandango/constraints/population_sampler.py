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

A final verification gate re-checks the assembled batch in every case, and per-tree hard
constraints from the spec are co-enforced -- every constructed individual is drawn to satisfy them
(the candidate source rejection-fuzzes). Everything else raises a clear error rather than silently
degrading: multiple population requirements at once and multi-valued inner fields (the
grouping-policy case) are future work.
"""

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
        if len(self.requirements) > 1:
            raise NotImplementedError(
                f"v1 supports a single population requirement; got {len(self.requirements)}. "
                f"Jointly satisfying multiple requirements is future work."
            )
        req = self.requirements[0]
        if req.operator not in _CONSTRUCTIBLE_OPERATORS:
            raise NotImplementedError(
                f"Operator '{req.operator.value}' is not supported for a population requirement."
            )
        reducer = req.aggregate.reducer_name
        if reducer == "fraction":
            return self._sample_fraction(req, n)
        if reducer == "distinct_count":
            return self._sample_distinct(req, n)
        if reducer in _DISTRIBUTION_QUANTILES:
            return self._sample_distribution(req, n)
        raise NotImplementedError(
            f"v1 constructs `fraction` quotas, `distinct_count` diversity, and distributional "
            f"fits ({', '.join(sorted(_DISTRIBUTION_QUANTILES))}); got '{reducer}'."
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

    # -- fraction quota construction ---------------------------------------- #
    def _sample_fraction(self, req: PopulationRequirement, n: int) -> list[DerivationTree]:
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

        batch = true_bucket + false_bucket
        random.shuffle(batch)
        self._verify(req, inner, batch, target_true)
        return batch

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

    # -- distinct-value diversity construction ------------------------------ #
    def _sample_distinct(self, req: PopulationRequirement, n: int) -> list[DerivationTree]:
        target, mode = self._target_distinct(req, n)
        inner = self._inner(req)

        # `>= 0` (or `< 1` -> handled as infeasible) imposes nothing; just fuzz a plain batch.
        if target <= 0:
            batch = [self._candidate() for _ in range(n)]
            self._verify_distinct(req, inner, batch)
            return batch

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
        batch: list[DerivationTree] = list(rep_trees)
        # Fill the remaining slots by reusing representatives (deep-copied so each batch entry is
        # an independent tree), cycling through them -- this keeps the distinct count exactly
        # len(reps) without re-fuzzing, so a high-cardinality grammar can't cause a false shortfall.
        i = 0
        while len(batch) < n:
            src = rep_trees[i % len(rep_trees)]
            batch.append(src.deepcopy(copy_parent=False))
            i += 1

        random.shuffle(batch)
        self._verify_distinct(req, inner, batch)
        return batch

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

    # -- distributional-shape construction ---------------------------------- #
    def _sample_distribution(
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
        if req.operator not in (Comparison.LESS_EQUAL, Comparison.LESS):
            raise NotImplementedError(
                f"A distributional fit is a distance to a target; only `<=`/`<` are meaningful "
                f"(match within delta). Got '{req.operator.value}' for '{reducer}'."
            )
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
        batch: list[DerivationTree] = []
        for i in range(n):
            q = quantile((i + 0.5) / n)
            j = bisect.bisect_left(pool_values, q)
            nearest = min(
                (k for k in (j - 1, j, j + 1) if 0 <= k < len(pool)),
                key=lambda k: abs(pool_values[k] - q),
            )
            batch.append(pool[nearest][1].deepcopy(copy_parent=False))

        random.shuffle(batch)
        self._verify_fit(req, inner, batch)
        return batch

    def _verify_fit(
        self, req: PopulationRequirement, inner: _InnerValue, batch: list[DerivationTree]
    ) -> None:
        """Independent gate: the assembled batch's distributional fit must satisfy the operator.
        A coarse grammar may be unable to reach ``delta``; that surfaces here as a shortfall."""
        values = [self._sole_value(inner, tree) for tree in batch]
        actual = REDUCERS[req.aggregate.reducer_name](values, *req.aggregate.reducer_args)
        if not req.operator.compare(actual, req.bound):
            raise PopulationShortfallError(
                f"Verification gate failed for '{req.aggregate.reducer_name} "
                f"{req.operator.value} {req.bound}': assembled batch has fit {actual:.4f}. The "
                f"grammar may be too coarse to match the target distribution that closely."
            )
