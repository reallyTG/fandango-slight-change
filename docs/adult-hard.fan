# adult-hard.fan — hard population-level requirements (Mechanism B), RUNNABLE.
#
# A population `where` (any `where` that mentions the reserved `population` binder) is a
# guarantee over the whole emitted batch of N, constructed by the sampler above the GA
# rather than steered. Run it with a fixed batch size, e.g.:
#
#   fandango fuzz -f docs/adult-hard.fan -n 200
#
# A label-encoded slice of the Adult census. Population requirements must target DISJOINT
# fields, so each column below is constrained by at most one population requirement.

<start> ::= <age> "," <occupation> "," <education_num> "," <hours> "," <income> "\n"

# age 10-99 (integer); occupation is 14 label-encoded categories;
# education_num 1-16; hours 1-99.
<age>           ::= <non_zero> <digit>
<occupation>    ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "13"
<education_num> ::= <non_zero> | "1" <digit> | "16"
<hours>         ::= <non_zero> <digit>
<income>        ::= "0" | "1"

<non_zero> ::= "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
<digit>    ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

# --- per-record hard validity (Fandango-native, co-enforced by the sampler) ---
# Use simple comparisons: the sampler rejection-fuzzes each candidate against these. (A
# single chained `17 <= int(<age>) <= 90` is not co-enforced by the sampler in v1.)
where int(<age>) >= 17
where int(<age>) <= 90

# --- population-level HARD requirements (Mechanism B) ---

# (a) marginal quota — exact by construction. 24% of records have income==1.
#     (An `== p` where p*N is non-integral snaps to the nearest achievable count and warns.)
where fraction(int(<income>) == 1 for x in population) == 0.24

# (a) count/diversity — at least 12 of the 14 occupations appear across the batch.
where distinct_count(<occupation> for x in population) >= 12

# (b) continuous shape — within δ by construction. `normal_fit(..., mu, sigma) <= δ` is the
#     1-Wasserstein distance to Normal(38, 13); only `<=`/`<` are meaningful (a distance).
#     δ below the discretization floor (~0.25 for an integer field) is rejected as
#     unsatisfiable in principle, so keep δ above it.
where normal_fit([int(<age>) for x in population], 38, 13) <= 0.7

# (c) coupling — a specific correlation between two fields, per record. `>=`/`>`/`<=`/`<`
#     drive toward the +/-1 extreme; `== r` targets r via a copula search (within a tolerance,
#     since exact Pearson equality is unachievable on discrete values).
where correlation((int(<education_num>), int(<hours>)) for x in population) >= 0.3


# ===========================================================================
# FUTURE WORK — not yet constructible; shown for orientation, kept commented.
# ===========================================================================
#
# Conditional P(y | x) — "y distributed according to x" — is an irreducible per-row
# coupling (x and y drawn together per record). v1 has no per-row `:=` seam (Q2), so the
# supported path is the *umbrella-symbol* pattern: emit the whole correlated tuple from one
# symbol's `:=` generator, e.g.
#
#   <age_salary> ::= <a> "," <s> := draw_age_and_salary()
#
# so (age, salary) is coupled by construction and the grammar never draws them apart.
#
# Also future work: grouped/stratified quotas (`P(income=1 | age_bucket)`), >2-way coupling,
# and the subset-selection path where the bootstrap equivalence CI in statistics/equivalence.py
# does real inferential work (on the by-construction path here, the gate is the point estimate).
