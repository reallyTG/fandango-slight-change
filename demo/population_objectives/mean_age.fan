# Sanity-check spec: steer the MEAN of the <age> column toward 30.
# A 5-row table of ages; the soft population objective pulls the population's
# average age toward 30. Drop the `minimizing` line to get the baseline.
<start>  ::= <row> "\n" <row> "\n" <row> "\n" <row> "\n" <row>
<row>    ::= <age>
<age>    ::= <digit> <digit>
<digit>  ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

minimizing abs(mean(int(<age>) for x in population) - 30)
