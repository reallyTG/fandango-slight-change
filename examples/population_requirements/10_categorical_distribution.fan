# Example 10 — categorical distribution over one field (hard, by construction)
#
# Several `fraction(...) == p` requirements on the SAME field are combined into one categorical
# distribution over that field: the sampler buckets fuzzed records by which cell each falls into and
# assembles the exact per-value counts. (A lone `fraction` line is still a single quota — Example 1;
# it takes two or more same-field cells to make a distribution.)
#
# The cells must be MUTUALLY EXCLUSIVE (each record's value belongs to exactly one) — `== 1`, `== 2`,
# ... are; a record satisfying two cells (e.g. `== 1` and `< 3`) is a spec error, raised loudly.
#
# The shares may sum to EXACTLY 1 (a full partition, below) or to LESS than 1 (the leftover is a
# free "anything else" remainder). Summing to more than 1 is rejected. As with a single quota,
# `p * N` must be integral to be hit exactly; at an awkward N the counts snap (largest-remainder, so
# they still sum to N) and a warning names the effective share.
#
# Run (of 20 records: exactly 4 ones, 8 twos, 2 threes, 6 fours):
#   PYTHONPATH=src fandango fuzz -f examples/population_requirements/10_categorical_distribution.fan -n 20
#
# Also try a PARTIAL partition — drop the last two lines: 10 ones + 4 twos, remaining 6 are any value.

<start> ::= <cat> "\n"
<cat>   ::= "1" | "2" | "3" | "4"

where fraction(int(<cat>) == 1 for x in population) == 0.2
where fraction(int(<cat>) == 2 for x in population) == 0.4
where fraction(int(<cat>) == 3 for x in population) == 0.1
where fraction(int(<cat>) == 4 for x in population) == 0.3
