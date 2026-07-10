# Example 1 — steer a column's MEAN toward a target (location objective)
#
# A small CSV of (name, age) rows. The soft population objective pulls the
# AVERAGE age across the whole generated population toward 40.
#
# This is the case soft population objectives handle *well*: they reliably steer
# where a distribution sits (its centre). Delete the `minimizing` line to see the
# unsteered baseline.
#
# Run:
#   PYTHONPATH=src fandango fuzz -f examples/population_objectives/01_mean_age.fan -n 20

<start>  ::= <row> ("\n" <row>)*
<row>    ::= <name> "," <age>
<name>   ::= <letter>{3,8}
<letter> ::= "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h"
<age>    ::= <digit> <digit>
<digit>  ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

# `population` is a reserved binder: the generator ranges over every record in the
# generated set, and `mean(...)` reduces the per-row values into one aggregate.
minimizing abs(mean(int(<age>) for x in population) - 40)
