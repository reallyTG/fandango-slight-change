# adult-hard.fan — ILLUSTRATIVE companion to PLAN-population-hard-constraints.md
#
# NOT YET RUNNABLE. Demonstrates the proposed Mechanism B surface syntax
# (`requiring … == …`, `requiring … >= …`, `requiring … within δ at 1-α`) on the
# label-encoded Adult census grammar. See PLAN-population-hard-constraints.md §5.

<start> ::= <age> <comma> <workclass> <comma> <fnlwgt> <comma> <education> <comma>
            <education_num> <comma> <marital_status> <comma> <occupation> <comma>
            <relationship> <comma> <race> <comma> <sex> <comma> <capital_gain> <comma>
            <capital_loss> <comma> <hours_per_week> <comma> <native_country> <comma> <income>

<age>        ::= <non_zero> <digit>                       # 10–99, integer
<occupation> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7"
               | "8" | "9" | "10" | "11" | "12" | "13"    # 14 categories
<income>     ::= "0" | "1"
# … workclass, fnlwgt, education, … native_country as in adult.fan …

<comma>    ::= ","
<non_zero> ::= "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
<digit>    ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

# --- per-record hard validity (unchanged, Fandango-native) ---
where 17 <= int(<age>) <= 90

# --- population-level HARD requirements (Mechanism B) ---

# (a) marginal quota — exact by construction
#     -> quota vector {income=0: 0.76*N, income=1: 0.24*N}; one GA job per cell with an
#        injected `where int(<income>) == k`, concatenated.
requiring fraction(int(<income>) == 1 for x in population) == 0.24

# (a) count-family — at least 12 of the 14 occupations must appear across the batch
#     -> coverage constraint on the plan; exact integer check at the gate.
requiring distinct_count(<occupation> for x in population) >= 12

# (b) continuous shape — equivalence-tested to a tolerance
#     within 0.5 = δ (max Wasserstein), at 0.95 = confidence 1-α.
#     `within 0.1` would be REJECTED at parse time: below the integer-rounding floor (~0.25).
requiring normal(int(<age>) for x in population, mu=38, sigma=13) within 0.5 at 0.95


# ===========================================================================
# "By construction" variant for the continuous column (sample-then-structure)
# ===========================================================================
# Replace the <age> rule above with a `:=` generator hook so ages are DRAWN from the
# target instead of grammar-random. The `requiring normal(...)` clause then degrades to a
# cheap regression check that structuring didn't perturb the draw (PLAN §4.4, §8).
#
# <age> ::= <non_zero> <digit> := str(next_age())
#
# def _make_age_pool(n, mu=38, sigma=13, lo=17, hi=90):
#     from statistics import NormalDist
#     nd = NormalDist(mu, sigma)
#     # inverse-CDF draw, clamped to the grammar's valid integer range, rounded to int
#     return iter(sorted(min(hi, max(lo, round(nd.inv_cdf((i + 0.5) / n))))
#                        for i in range(n)))
#
# _age_pool = None
# def next_age():
#     global _age_pool
#     if _age_pool is None:
#         _age_pool = _make_age_pool(POPULATION_N)   # N injected by the sampler
#     return next(_age_pool)
#
# Open items (PLAN §12): the age pool must be split across the income-quota jobs, and the
# `:=`-pinned age must be repair-exempt (simple.py:470) or the gate will reject perturbed rows.


# ===========================================================================
# Conditional / joint requirements — "y distributed according to x" (PLAN §3c)
# ===========================================================================
# A conditional P(y | x) is an irreducible per-row coupling: x and y are drawn TOGETHER
# per record. It does not decompose into independent per-field jobs. Two sub-cases:

# --- Case 1: discrete/bucketable conditioning -> nested quota (exact by construction) ---
# P(income=1 | age) supplied as a per-bucket curve; stratify by age band, enforce the
# income fraction within each band.
#
# requiring fraction(int(<income>) == 1 for x in population) == p_high(int(<age>))
#          grouped by age_bucket(int(<age>))
#
# def age_bucket(a):
#     return min(a // 10, 6)                         # decade buckets
# def p_high(a):
#     return {2: 0.05, 3: 0.20, 4: 0.35, 5: 0.42, 6: 0.28}.get(age_bucket(a), 0.10)

# --- Case 2: continuous conditional shape -> per-row coupled draw ---
# Hypothetical <salary>; salary | age ~ LogNormal(mu(age), sigma). Salary's generator reads
# the row's ALREADY-DRAWN age, so (age, salary) is coupled by construction.
#
# <salary> ::= <digit>{1,6} := str(draw_salary_given(int(<age>)))
#
# def draw_salary_given(age):
#     from statistics import NormalDist
#     mu = 9.5 + 0.02 * age                          # log-salary grows with age
#     return round(math.exp(NormalDist(mu, 0.4).inv_cdf(row_uniform())))
#
# Verify the conditional (cannot check salary's marginal alone), cheapest mode first:
#   (i) stratified 1-D equivalence — salary|bucket passes the gate in each band:
#   requiring lognormal(int(<salary>) for x in population, mu=mu_of_age, sigma=0.4)
#            within 0.5 at 0.95 grouped by age_bucket(int(<age>))
#   (ii) dependence-only — reuse Mechanism A's joint correlation reducer (population.py:67):
#   requiring correlation((int(<age>), int(<salary>)) for x in population) within 0.05 at 0.95
#   (iii) full joint — 2-D energy distance / joint-Wasserstein (most faithful, most expensive).
#
# Open items (PLAN §12): the generator resolver must draw <age> BEFORE draw_salary_given(age)
# runs (age->salary direction), and feasibility/shortfall is reported per (bucket, outcome) cell.
