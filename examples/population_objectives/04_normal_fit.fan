# Example 4 — fit a column's SHAPE to a target distribution
#
# The objective minimises the 1-Wasserstein distance between the population's
# <age> values and a Normal(30, 5) target, pulling both the centre (~30) and,
# in principle, the spread (~5).
#
# HONEST CAVEAT (read before relying on this):
#   Soft objectives steer LOCATION (the mean) reliably, but do NOT reliably
#   steer SPREAD. Selection acts on whole records, and within-record dispersion
#   isn't attributable to any single record, so `normal_fit` mostly re-centres
#   the column rather than tightening its variance. If you need a distribution
#   matched *as a requirement* (shape included), that's the sample-then-structure
#   / generator-hook direction, not this soft objective. See docs/Distributions.md.
#
# Run:
#   PYTHONPATH=src fandango fuzz -f examples/population_objectives/04_normal_fit.fan -n 20

<start> ::= <row> ("\n" <row>)*
<row>   ::= <age>
<age>   ::= <digit> <digit>
<digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

# normal_fit(values, mu, sigma) — note the list-comprehension brackets around the
# per-row values; the fit reducers take an explicit sequence.
minimizing normal_fit([int(<age>) for x in population], 30, 5)
