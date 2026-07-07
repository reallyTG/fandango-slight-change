<start> ::= <person> "\n" <person> "\n" <person>
<person> ::= <age>
<age> ::= <digit> <digit>
<digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

# Soft population-level objective (max goal): push the mean of all <age> values up.
maximizing mean(int(<age>) for x in population)
