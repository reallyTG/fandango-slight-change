# Sanity-check spec: steer the SHAPE of the <age> column toward Normal(30, 5).
# Same 5-row table; the soft objective minimises the 1-Wasserstein distance from
# the population's ages to N(30, 5), pulling both centre (~30) and spread (~5).
<start>  ::= <row> "\n" <row> "\n" <row> "\n" <row> "\n" <row>
<row>    ::= <age>
<age>    ::= <digit> <digit>
<digit>  ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

minimizing normal_fit([int(<age>) for x in population], 30, 5)
