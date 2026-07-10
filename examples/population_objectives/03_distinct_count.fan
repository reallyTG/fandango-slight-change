# Example 3 — maximise VARIETY across the population (coverage / diversity)
#
# A one-column table of city-like tokens. The objective maximises the number of
# DISTINCT values across the generated set, so the batch spreads across many
# categories instead of repeating a few.
#
# `distinct_count` is a between-record objective, so selection can steer it well
# (unlike within-record spread — see Example 4's note).
#
# Run:
#   PYTHONPATH=src fandango fuzz -f examples/population_objectives/03_distinct_count.fan -n 20

<start>  ::= <row> ("\n" <row>)*
<row>    ::= <city>
<city>   ::= <letter>{3,6}
<letter> ::= "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j"

maximizing distinct_count(<city> for x in population)
