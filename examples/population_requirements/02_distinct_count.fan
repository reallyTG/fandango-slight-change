# Example 2 — distinct-value diversity (hard, by construction)
#
# `distinct_count(<field> for x in population) OP K` guarantees the number of DISTINCT
# values of a field across the batch. The sampler gathers one representative per distinct
# value until the target is reached (for `>=`/`>`/`==`) or capped (for `<=`/`<`), then
# fills the remaining slots by reusing representatives so the distinct count lands exactly
# on target — no re-fuzzing that might drift the count.
#
# Here: at least 4 of the 5 possible occupations must appear across the batch.
#
# Run:
#   PYTHONPATH=src fandango fuzz -f examples/population_requirements/02_distinct_count.fan -n 20
#
# Feasibility: `>= K` needs the grammar to *have* K distinct values AND N >= K. Asking for
# more distinct values than the grammar can produce, or than N, is a clear shortfall.
# Try `== 5` (all five) or `<= 2` (cap the variety).

<start> ::= <occupation> "\n"
<occupation> ::= "engineer" | "doctor" | "artist" | "lawyer" | "teacher"

where distinct_count(<occupation> for x in population) >= 4
