# Example 5 — combine several requirements + per-record constraints (hard)
#
# Multiple population requirements are enforced together when they target DISJOINT fields
# (each field constrained by at most one requirement). The sampler plans a column per
# requirement and grafts each field into shared skeleton records, so the requirements don't
# interfere and every requirement's gate holds independently. Same-field / nested-field sets
# are rejected with a clear error.
#
# Per-record `where` constraints (no `population` binder) are HARD validity rules, co-enforced
# alongside the population requirements: every constructed record satisfies them (the sampler
# rejection-fuzzes its candidate source). Use simple comparisons — a single chained
# `18 <= int(<age>) <= 65` is not co-enforced by the sampler; split it into two lines.
#
# This batch simultaneously guarantees:
#   - every record has 18 <= age <= 65        (per-record hard validity)
#   - exactly 40% of records are "premium"    (fraction quota)
#   - at least 3 distinct regions appear      (diversity)
#
# Run:
#   PYTHONPATH=src fandango fuzz -f examples/population_requirements/05_combined.fan -n 50

<start> ::= <age> "," <tier> "," <region> "\n"
<age>    ::= <d> <d>
<tier>   ::= "basic" | "premium"
<region> ::= "N" | "S" | "E" | "W"
<d>      ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

# per-record validity (co-enforced; simple comparisons, not a chained one)
where int(<age>) >= 18
where int(<age>) <= 65

# population-level guarantees on disjoint fields
where fraction(<tier> == "premium" for x in population) == 0.40
where distinct_count(<region> for x in population) >= 3
