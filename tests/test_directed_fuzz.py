import random
import unittest

import fandango.language.grammar.nodes as nodes
from fandango.errors import FandangoValueError
from fandango.language.grammar import directed
from fandango.language.parse.parse import parse
from fandango.language.symbols import NonTerminal

# Each alternative carries a marker terminal that only it produces.
STATEMENTS = """
<start> ::= <stmt>+
<stmt> ::= <expr> ";" | "if" <expr> "{" <stmt>* "}" | "while" <expr> "{" <stmt>+ "}"
<expr> ::= <num> | <expr> "+" <expr> | "(" <expr> ")" | "f" ("@" | "#" <num>)
<num> ::= "1" | "2"
<orphan> ::= "never"
"""
MARKERS = {0: ";", 1: "if", 2: "while"}

# The target sits six rules deep and the normal fuzzer prefers the exits.
CHAIN = """
<start> ::= "x" | "x" | "x" | <l1>
<l1> ::= "y" | "y" | "y" | <l2>
<l2> ::= "y" | "y" | "y" | <l3>
<l3> ::= "y" | "y" | "y" | <l4>
<l4> ::= "y" | "y" | "y" | <l5>
<l5> ::= "y" | "deep"
"""


def load(spec):
    grammar, _ = parse(spec, use_stdlib=False, use_cache=False)
    return grammar


class TestDirectedFuzz(unittest.TestCase):
    def setUp(self):
        nodes.MAX_REPETITIONS = 20
        random.seed(7)
        self.grammar = load(STATEMENTS)

    def test_every_alternative_is_used(self):
        for shortest in (True, False):
            for index, marker in MARKERS.items():
                for _ in range(30):
                    tree = self.grammar.fuzz(
                        require=("<stmt>", index), max_nodes=30, shortest=shortest
                    )
                    text = str(tree)
                    self.assertIn(marker, text)
                    self.assertIsNotNone(self.grammar.parse(text))

    def test_nested_alternative(self):
        branches = directed.alternative_branches(self.grammar, "<expr>")
        specs = [b.format_as_spec() for b in branches]
        hash_branch = branches[specs.index("'#' <num>")]
        for _ in range(30):
            text = str(self.grammar.fuzz(require=hash_branch, max_nodes=30))
            self.assertIn("f#", text)

    def test_shortest_path_is_minimal(self):
        tree = self.grammar.fuzz(require=("<stmt>", 2), max_nodes=0)
        # while <num> { <num> ; } is the smallest program with a while.
        self.assertRegex(str(tree), r"^while[12]\{[12];\}$")

    def test_rare_target(self):
        grammar = load(CHAIN)
        plain = sum("deep" in str(grammar.fuzz(max_nodes=50)) for _ in range(200))
        self.assertLess(plain, 20)
        target = directed.alternatives(grammar, "<l5>")[1]
        for shortest in (True, False):
            for _ in range(50):
                tree = grammar.fuzz(require=target, max_nodes=50, shortest=shortest)
                self.assertEqual(str(tree), "deep")

    def test_seed_determinism(self):
        def sample():
            random.seed(3)
            return [
                str(
                    self.grammar.fuzz(
                        require=("<stmt>", 1), max_nodes=40, shortest=False
                    )
                )
                for _ in range(10)
            ]

        self.assertEqual(sample(), sample())

    def test_size_tracks_max_nodes(self):
        for max_nodes in (20, 60):
            sizes = [
                self.grammar.fuzz(
                    require=("<stmt>", 2), max_nodes=max_nodes, shortest=False
                ).size()
                for _ in range(100)
            ]
            self.assertLess(sorted(sizes)[50], 2 * max_nodes)

    def test_unreachable(self):
        with self.assertRaises(FandangoValueError):
            self.grammar.fuzz(require=("<orphan>", 0))
        with self.assertRaises(FandangoValueError):
            self.grammar.fuzz(require=("<stmt>", 3))
        batch = directed.fuzz_alternatives(self.grammar, ["<orphan>"], 3)
        self.assertEqual(batch, {(NonTerminal("<orphan>"), 0): []})

    def test_batch(self):
        batch = directed.fuzz_alternatives(
            self.grammar, ["<stmt>", "<expr>"], 5, max_nodes=30, nested=True
        )
        self.assertEqual(len(batch), 3 + 6)
        self.assertTrue(all(len(trees) == 5 for trees in batch.values()))
        for (_rule, index), trees in batch.items():
            if _rule == NonTerminal("<stmt>"):
                for tree in trees:
                    self.assertIn(MARKERS[index], str(tree))


if __name__ == "__main__":
    unittest.main()
