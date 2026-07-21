# Example 1 — exact fraction quota (hard, by construction)
#
# Mechanism B: any `where` that mentions the reserved `population` binder is a HARD
# requirement over the whole emitted batch of N, constructed by the sampler (not steered
# by the GA). Run it with a fixed batch size.
#
# `fraction(<predicate> for x in population)` is the share of records satisfying the
# predicate. `== 0.30` guarantees EXACTLY 30% by construction — the sampler buckets
# fuzzed records by the predicate and assembles the exact count.
#
# Snap-and-warn: `0.30 * N` must be an integer to be hit exactly. At an awkward N (say
# N = 7, 0.30*7 = 2.1) the sampler snaps to the nearest achievable count (2/7) and prints
# a warning naming the effective target — a hard requirement can't invent a fractional row.
#
# Run (30% of 20 = exactly 6 records with income == 1):
#   PYTHONPATH=src fandango fuzz -f examples/population_requirements/01_fraction_quota.fan -n 20
#
# Also try `>= 0.30` (minimal satisfying count, 6/20) or `<= 0.25` (maximal, 5/20).

<start>  ::= <income> "\n"
<income> ::= "0" | "1"

where fraction(int(<income>) == 1 for x in population) == 0.30
