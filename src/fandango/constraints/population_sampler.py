"""Mechanism B: *constructing* a batch that satisfies a hard population requirement.

Where the GA steers a per-tree fitness (best-effort), a hard population ``where`` is a
guarantee over the whole emitted batch of N -- something no single individual can satisfy and
that streaming, per-tree evolution cannot express. This sampler sits *above* the GA and
constructs such a batch directly.

v1 covers the exact-by-construction **quota** case: a single
``fraction(<pred> for x in population) OP p`` requirement whose inner element yields exactly one
boolean per individual. It buckets freshly-fuzzed individuals by the predicate and assembles the
target count, so the resulting batch hits the fraction exactly (for ``==``, snapped to the
nearest achievable ``round(p*N)/N``) or the operator's boundary (for the inequalities). A final
verification gate re-checks the assembled batch.

Everything outside that slice raises a clear error rather than silently degrading:
multiple requirements, non-``fraction`` reducers (distributional fits, ``distinct_count``
diversity), multi-valued inner fields (the grouping-policy case), and per-tree hard constraints
alongside a requirement are all future work.
"""

import math
import random
from typing import Any, Optional

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

# Operators the quota path can construct toward. ``!=`` is intentionally excluded: "not exactly
# 30%" is an odd guarantee and has no natural construction target.
_QUOTA_OPERATORS = frozenset(
    {
        Comparison.EQUAL,
        Comparison.GREATER_EQUAL,
        Comparison.LESS_EQUAL,
        Comparison.GREATER,
        Comparison.LESS,
    }
)


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
        max_attempts_per_slot: int = 1000,
    ) -> None:
        self.grammar = grammar
        self.requirements = (
            list(requirements)
            if requirements is not None
            else list(getattr(grammar, "population_requirements", []))
        )
        self._max_attempts_per_slot = max(1, max_attempts_per_slot)
        # The environment the inner predicate is eval'd in (so spec-level `def`s/imports work).
        self._global_variables, self._local_variables = grammar.get_spec_env()

    def sample(self, n: int) -> list[DerivationTree]:
        """Return a batch of ``n`` individuals satisfying the population requirements."""
        if n <= 0:
            raise FandangoValueError(f"Population size must be positive; got {n}.")
        if not self.requirements:
            return [self.grammar.fuzz() for _ in range(n)]
        if len(self.requirements) > 1:
            raise NotImplementedError(
                f"v1 supports a single population requirement; got {len(self.requirements)}. "
                f"Jointly satisfying multiple requirements is future work."
            )
        return self._sample_quota(self.requirements[0], n)

    # -- quota construction ------------------------------------------------- #
    def _sample_quota(self, req: PopulationRequirement, n: int) -> list[DerivationTree]:
        reducer = req.aggregate.reducer_name
        if reducer != "fraction":
            raise NotImplementedError(
                f"v1 construction supports only the `fraction` quota; got '{reducer}'. "
                f"Distributional fits and diversity (`distinct_count`) requirements are "
                f"future work."
            )
        if req.operator not in _QUOTA_OPERATORS:
            raise NotImplementedError(
                f"Operator '{req.operator.value}' is not supported for a fraction quota."
            )

        target_true = self._target_count(req, n)
        target_false = n - target_true
        inner = _InnerValue(
            req.aggregate.inner_expression,
            searches=req.aggregate.inner_searches,
            global_variables=self._global_variables,
            local_variables=self._local_variables,
        )

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
            tree = self.grammar.fuzz()
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
        """Evaluate the inner boolean predicate against one individual.

        v1 requires exactly one inner value per tree (each individual = one record). A tree
        yielding several values is the pooled/multiplicity case (grouping policy) and is not
        yet supported.
        """
        values = inner.raw_values(tree)
        if len(values) != 1:
            raise NotImplementedError(
                f"v1 requires exactly one inner value per individual, but one tree yielded "
                f"{len(values)}. Multi-valued fields need the grouping policy (future work)."
            )
        return bool(values[0])

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
