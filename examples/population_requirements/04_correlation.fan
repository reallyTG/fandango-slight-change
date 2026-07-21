# Example 4 — couple two columns at a target correlation (hard, by construction)
#
# `correlation((<x>, <y>) for x in population) OP r` constructs N (x, y) pairs whose Pearson
# correlation meets the bound, grafting both fields together per record so the coupling
# holds within each individual.
#
# Two modes:
#   * Inequalities (`>=`/`>`/`<=`/`<`): pair the fields monotonically (toward +1) or
#     anti-monotonically (toward -1) to reach the bound.
#   * Exact `== r`: target a SPECIFIC correlation via a Gaussian-copula rank pairing,
#     searched over many draws for the ranking whose achieved correlation is nearest r.
#     Exact Pearson equality is unachievable on discrete values, so the gate is a tolerance
#     band (`correlation_tolerance`, default 0.15) — the correlation analogue of the
#     fraction snap-and-warn.
#
# Run (age and score positively coupled, r ~ 0.6):
#   PYTHONPATH=src fandango fuzz -f examples/population_requirements/04_correlation.fan -n 100
#
# Change `== 0.6` to `>= 0.9` (drive toward the max) or `== -0.5` (anti-correlated).
# Note: the two fields must be DISJOINT and each read exactly one field of the tuple.

<start> ::= <age> "," <score> "\n"
<age>   ::= <d> <d>
<score> ::= <d> <d>
<d>     ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

where correlation((int(<age>), int(<score>)) for x in population) == 0.6
