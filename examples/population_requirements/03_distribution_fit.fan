# Example 3 — match a distribution shape within a tolerance (hard, by construction)
#
# `normal_fit([<x> for x in population], mu, sigma) <= delta` guarantees the <x> column's
# distribution is within `delta` (a 1-Wasserstein / earth-mover distance) of Normal(mu,
# sigma). The sampler fuzzes a candidate pool and places each order-statistic slot on the
# target quantile Q((i+0.5)/N), so the batch's shape matches as closely as the grammar's
# own values allow. Only `<=`/`<` are meaningful — it's a distance to a target.
#
# Built-in fits: normal_fit, lognormal_fit, uniform_fit, exponential_fit. Add your own with
# register_distribution_fit (just supply a quantile function) — see docs/Distributions.md.
#
# The discretization floor (important): a continuous target realized on an INTEGER field
# sits a fixed distance (~0.25 for unit steps) from the true continuous distribution no
# matter how values are placed. A `delta` below that floor is unsatisfiable *in principle*,
# and the sampler says so precisely. Here <age> steps by 1, so keep delta above ~0.25.
# Try `<= 0.1` to see the floor diagnosis.
#
# Run (age column matched to Normal(30, 8) within 0.5):
#   PYTHONPATH=src fandango fuzz -f examples/population_requirements/03_distribution_fit.fan -n 100

<start> ::= <age> "\n"
<age>   ::= <d> <d>
<d>     ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

where normal_fit([int(<age>) for x in population], 30, 8) <= 0.5
