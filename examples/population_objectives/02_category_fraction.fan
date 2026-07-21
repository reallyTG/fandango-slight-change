# Example 2 — steer the FRACTION of records in a category (marginal proportion)
#
# A one-column table of a categorical <sex> field. The objective pushes the
# population toward ~half "F". Change the target expression to bias the mix
# (e.g. `- 0.3` inside an `abs(... )` to aim for 30% F).
#
# `fraction(<predicate> for x in population)` = share of records satisfying the
# predicate. This is a soft *bias*, not a hard quota — for an exact proportion
# guarantee use a hard population `where` (Mechanism B), e.g.
# `where fraction(<sex> == "F" for x in population) == 0.3`.
#
# Run:
#   PYTHONPATH=src fandango fuzz -f examples/population_objectives/02_category_fraction.fan -n 20

<start> ::= <row> ("\n" <row>)*
<row>   ::= <sex>
<sex>   ::= "M" | "F"

# Aim for a balanced 50/50 split: maximise how close the F-fraction is to 0.5
# by minimising the absolute gap.
minimizing abs(fraction(<sex> == "F" for x in population) - 0.5)
