"""Equivalence-test statistics for hard continuous population requirements.

A continuous distribution requirement ("age ~ N(30, 5)") can never be proven *exactly*
from a finite sample. The honest contract is an **equivalence test**: bound the divergence
between the emitted column and the target below a tolerance ``delta`` with confidence
``1 - alpha``. A goodness-of-fit null ("data came from the target") is the wrong tool —
failing to reject is not evidence of equivalence, small N passes anything, and large N
rejects any deviation, so a constructed batch gets *harder* to pass as it grows. An
equivalence test puts the burden of proof on demonstrating closeness instead.

This module is the pure, GA-free statistical core:

* :func:`wasserstein_distance` — the point-estimate 1-Wasserstein distance to a target
  (the same metric the soft ``wasserstein_fit`` reducer uses; kept here as a pure helper).
* :func:`wasserstein_upper_ci` — a bootstrap ``1 - alpha`` upper confidence bound on that
  distance. Accept equivalence iff this bound is below ``delta``.
* :func:`ks_upper_ci` — the same idea with the Kolmogorov–Smirnov statistic, for matching
  an observer whose likely test is a KS goodness-of-fit.
* :func:`discretization_floor` — the smallest Wasserstein distance a *rounded* field can
  reach from a continuous target. Below it, a ``delta`` is unsatisfiable in principle.
* :func:`familywise_alpha` — Bonferroni (or none) correction for a bundle of ``k`` tests.

Everything takes an explicit ``seed`` and uses a private RNG, so results are reproducible
without touching global ``random`` state and without the ``Date.now``/``random`` traps.
"""

import random as _random_module
from collections.abc import Callable, Sequence
from typing import Any, Optional


def wasserstein_distance(
    sample: Sequence[Any], target_quantile: Callable[[float], float]
) -> float:
    """1-Wasserstein (earth-mover) distance from ``sample`` to a target distribution
    described by its quantile function ``target_quantile(p)``, ``p`` in ``(0, 1)``.

    Each order statistic ``x_(i)`` is compared to ``target_quantile((i + 0.5) / n)`` and
    the mean absolute gap returned. Zero iff the sorted sample lies exactly on the target
    quantiles. Empty sample -> 0.0 (no evidence), never a ``ZeroDivisionError``.
    """
    xs = sorted(float(v) for v in sample)
    n = len(xs)
    if n == 0:
        return 0.0
    return sum(abs(x - target_quantile((i + 0.5) / n)) for i, x in enumerate(xs)) / n


def _empirical_upper_quantile(values: list[float], confidence: float) -> float:
    """The ``confidence`` empirical quantile of ``values`` (the upper CI bound).

    Uses the nearest-rank method on the sorted bootstrap statistics. ``confidence`` is the
    coverage of the one-sided bound (e.g. 0.95 -> the 95th percentile).
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if confidence <= 0:
        return ordered[0]
    if confidence >= 1:
        return ordered[-1]
    # nearest-rank: smallest value whose rank covers `confidence` of the mass
    rank = max(1, min(len(ordered), int(round(confidence * len(ordered)))))
    return ordered[rank - 1]


def _bootstrap_statistics(
    sample: Sequence[Any],
    statistic: Callable[[list[float]], float],
    *,
    n_boot: int,
    seed: int,
) -> list[float]:
    """Resample ``sample`` with replacement ``n_boot`` times, returning ``statistic`` of
    each resample. A private ``random.Random(seed)`` keeps this reproducible and isolated
    from global RNG state."""
    xs = [float(v) for v in sample]
    n = len(xs)
    if n == 0:
        return []
    rng = _random_module.Random(seed)
    stats = []
    for _ in range(n_boot):
        resample = [xs[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(resample))
    return stats


def wasserstein_upper_ci(
    sample: Sequence[Any],
    target_quantile: Callable[[float], float],
    *,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> float:
    """Bootstrap ``confidence`` (= ``1 - alpha``) upper confidence bound on the Wasserstein
    distance between ``sample`` and the target.

    Resamples ``sample`` with replacement, recomputes :func:`wasserstein_distance` on each
    resample, and returns the ``confidence`` percentile of that bootstrap distribution.
    Concluding equivalence means checking this bound ``< delta``: with probability
    ``confidence`` the true distance is no larger, so the closeness is *demonstrated*, not
    merely un-refuted. Empty sample -> 0.0.
    """
    stats = _bootstrap_statistics(
        sample,
        lambda rs: wasserstein_distance(rs, target_quantile),
        n_boot=n_boot,
        seed=seed,
    )
    return _empirical_upper_quantile(stats, confidence)


def ks_statistic(
    sample: Sequence[Any], target_cdf: Callable[[float], float]
) -> float:
    """Kolmogorov–Smirnov statistic: the largest gap between the empirical CDF of
    ``sample`` and the target CDF ``target_cdf``. In [0, 1]; empty sample -> 0.0."""
    xs = sorted(float(v) for v in sample)
    n = len(xs)
    if n == 0:
        return 0.0
    d = 0.0
    for i, x in enumerate(xs):
        cdf = target_cdf(x)
        # the empirical CDF jumps at each point; compare just below and just above
        d = max(d, abs(cdf - i / n), abs((i + 1) / n - cdf))
    return d


def ks_upper_ci(
    sample: Sequence[Any],
    target_cdf: Callable[[float], float],
    *,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> float:
    """Bootstrap ``confidence`` upper confidence bound on the KS statistic — the KS
    alternate to :func:`wasserstein_upper_ci`, for an observer likely to run a KS test.
    Empty sample -> 0.0."""
    stats = _bootstrap_statistics(
        sample,
        lambda rs: ks_statistic(rs, target_cdf),
        n_boot=n_boot,
        seed=seed,
    )
    return _empirical_upper_quantile(stats, confidence)


def _estimate_granularity(sample: Sequence[Any]) -> Optional[float]:
    """Estimate the quantization step of ``sample`` as the smallest positive gap between
    its sorted distinct values (a grid of integers -> 1.0; tenths -> 0.1). Returns ``None``
    when there are fewer than two distinct values (no grid to infer)."""
    distinct = sorted(set(float(v) for v in sample))
    if len(distinct) < 2:
        return None
    gaps = [b - a for a, b in zip(distinct, distinct[1:]) if b > a]
    return min(gaps) if gaps else None


def discretization_floor(
    sample: Optional[Sequence[Any]] = None, *, granularity: Optional[float] = None
) -> float:
    """The smallest 1-Wasserstein distance a field quantized to a fixed grid can reach from
    a *continuous* target.

    A continuous target rounded to a grid of step ``h`` incurs a quantization error that is
    approximately uniform on ``[-h/2, h/2]``, whose mean absolute value is ``h / 4``. So no
    matter how the values are placed, a grid-of-step-``h`` field sits at least ~``h / 4``
    from the continuous target. A ``delta`` below this floor is unsatisfiable in principle
    (see :func:`is_delta_below_floor`).

    Pass ``granularity`` explicitly, or a ``sample`` to infer it from the achieved values.
    Unknown granularity (constant sample, or neither argument) -> 0.0 (no floor claimed).
    """
    step = granularity
    if step is None and sample is not None:
        step = _estimate_granularity(sample)
    if step is None or step <= 0:
        return 0.0
    return step / 4.0


def is_delta_below_floor(
    delta: float,
    sample: Optional[Sequence[Any]] = None,
    *,
    granularity: Optional[float] = None,
    slack: float = 0.5,
) -> tuple[bool, float]:
    """Whether ``delta`` is below the discretization floor (hence unsatisfiable in
    principle), and the floor value.

    ``slack`` guards against declaring a requirement impossible on a floor *estimated* from
    a finite sample: only ``delta < slack * floor`` is flagged. With the default 0.5 a
    requirement is called impossible only when it asks for less than half the theoretical
    minimum — a clear, not a marginal, violation.
    """
    floor = discretization_floor(sample, granularity=granularity)
    return (floor > 0 and delta < slack * floor, floor)


def familywise_alpha(alpha: float, k: int, mode: str = "bonferroni") -> float:
    """Per-test significance level for a bundle of ``k`` requirements the user wants to hold
    *jointly* at level ``alpha``.

    ``bonferroni`` (default) splits the budget: ``alpha / k`` per test, guaranteeing the
    family-wise level. ``none`` applies ``alpha`` to each test (no correction; only correct
    when the tests are treated independently). ``k <= 1`` -> ``alpha`` unchanged.
    """
    if k <= 1:
        return alpha
    if mode == "none":
        return alpha
    if mode == "bonferroni":
        return alpha / k
    raise ValueError(f"Unknown family-wise mode {mode!r}; use 'bonferroni' or 'none'.")
