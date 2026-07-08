# Sanity-check spec: steer the JOINT relationship between two columns.
# A 5-row table of (age, income); the soft objective maximises the row-wise
# Pearson correlation between age and income across the population.
# NOTE: `correlation` pairs the two fields *per row* (see the joint-objective
# note in docs/Distributions.md); without that pairing the value would collapse
# to ~0 no matter what.
<start>   ::= <row> "\n" <row> "\n" <row> "\n" <row> "\n" <row>
<row>     ::= <age> "," <income>
<age>     ::= <digit> <digit>
<income>  ::= <digit> <digit>
<digit>   ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

maximizing correlation((int(<age>), int(<income>)) for x in population)
