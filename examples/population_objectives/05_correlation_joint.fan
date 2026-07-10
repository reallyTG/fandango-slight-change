# Example 5 — steer a JOINT relationship between two columns
#
# A two-column (age, income) table. The objective maximises the row-wise Pearson
# correlation between age and income across the population, so the two fields
# move together.
#
# KEY DETAIL: `correlation((int(<age>), int(<income>)) for x in population)` pairs
# the two fields *per row* (the tuple binds them within each record). Without that
# per-row pairing the correlation collapses to ~0 regardless of the data.
# Correlation is one of the objectives selection *can* move, because "all-high /
# all-low" records give it a between-record route.
#
# Run:
#   PYTHONPATH=src fandango fuzz -f examples/population_objectives/05_correlation_joint.fan -n 20

<start>  ::= <row> ("\n" <row>)*
<row>    ::= <age> "," <income>
<age>    ::= <digit> <digit>
<income> ::= <digit> <digit>
<digit>  ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

maximizing correlation((int(<age>), int(<income>)) for x in population)
