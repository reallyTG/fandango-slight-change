"""Directed generation: build a tree that is guaranteed to use one grammar node.

`fuzz_through(grammar, target)` returns a random derivation tree whose
derivation passes through `target`, a node of the grammar such as one
alternative of a rule. No search or constraint solving is involved:

1. `reach_costs` computes, for every grammar node, the size of the smallest
   subtree derivable from it that contains `target` (infinite if none). It is
   a fixpoint like `PrimerVisitor`, with `distance_to_completion` as the cost
   of the positions that do not carry the target.
2. Generation walks down from the start symbol. At an alternative it takes a
   branch that can reach the target; in a concatenation or repetition one
   position carries the target and the others are filled by the normal
   fuzzer. At `target` itself the normal fuzzer takes over.

With `shortest=True` every choice on the way takes the cheapest branch, so the
path is a shortest derivation path. With `shortest=False` a choice is random
among the branches whose cost fits the remaining node budget, as the normal
fuzzer chooses, which varies the context the target appears in.

Off-path positions get the same budget handling as `Concatenation.fuzz` and
`Repetition.fuzz`, so trees near `max_nodes` are completed at minimum depth.
`max_nodes` is a soft limit: if the target cannot be reached within it, the
cheapest path is taken anyway.

Nonterminals with a generator are not entered, because their subtree comes
from the generator and cannot be steered. A target reachable only through one
is reported as unreachable.

The target is identified by object identity. `alternatives(grammar, rule)`
returns a rule's top-level branches and `alternative_branches(grammar, rule)`
returns every branch of every alternative inside the rule, including inlined
ones such as `<a> ::= <b> (<c> | <d>)`.
"""

import random
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Union

from fandango.errors import FandangoValueError
from fandango.language.grammar.nodes.alternative import Alternative
from fandango.language.grammar.nodes.concatenation import Concatenation
from fandango.language.grammar.nodes.node import Node
from fandango.language.grammar.nodes.non_terminal import NonTerminalNode
from fandango.language.grammar.nodes.repetition import Repetition
from fandango.language.symbols import NonTerminal
from fandango.language.tree import DerivationTree

if TYPE_CHECKING:
    import fandango.language.grammar.grammar

INF = float("inf")
Cost = Union[int, float]

Requirement = Union[Node, tuple[Union[str, NonTerminal], int]]


def _nt(symbol: Union[str, NonTerminal]) -> NonTerminal:
    return NonTerminal(symbol) if isinstance(symbol, str) else symbol


def alternatives(
    grammar: "fandango.language.grammar.grammar.Grammar", rule: Union[str, NonTerminal]
) -> list[Node]:
    """The top-level branches of `rule`. A rule that is not an alternative has
    one branch, the rule itself."""
    nt = _nt(rule)
    if nt not in grammar.rules:
        raise FandangoValueError(f"Symbol {nt.format_as_spec()} not found in grammar")
    node = grammar.rules[nt]
    if isinstance(node, Alternative):
        return list(node.alternatives)
    return [node]


def alternative_branches(
    grammar: "fandango.language.grammar.grammar.Grammar", rule: Union[str, NonTerminal]
) -> list[Node]:
    """Every branch of every alternative inside `rule`, in preorder, without
    descending into other nonterminals. Includes the top-level branches."""
    nt = _nt(rule)
    if nt not in grammar.rules:
        raise FandangoValueError(f"Symbol {nt.format_as_spec()} not found in grammar")
    branches: list[Node] = []

    def walk(node: Node) -> None:
        if isinstance(node, NonTerminalNode):
            return
        if isinstance(node, Alternative):
            for branch in node.alternatives:
                branches.append(branch)
                walk(branch)
            return
        for child in node.children():
            walk(child)

    walk(grammar.rules[nt])
    return branches


def resolve(
    grammar: "fandango.language.grammar.grammar.Grammar", require: Requirement
) -> Node:
    """Turn `(rule, alt_index)` or a grammar node into the target node."""
    if isinstance(require, Node):
        return require
    rule, index = require
    branches = alternatives(grammar, rule)
    if not 0 <= index < len(branches):
        raise FandangoValueError(
            f"{_nt(rule).format_as_spec()} has {len(branches)} alternative(s); "
            f"index {index} is out of range"
        )
    return branches[index]


def _all_nodes(grammar: "fandango.language.grammar.grammar.Grammar") -> list[Node]:
    seen: set[int] = set()
    out: list[Node] = []
    stack: list[Node] = list(grammar.rules.values())
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        out.append(node)
        if not isinstance(node, NonTerminalNode):
            stack.extend(node.children())
    return out


def _carrier_costs(nodes: list[Node], costs: dict[int, Cost]) -> list[Cost]:
    """For a concatenation: the cost of carrying the target at each position."""
    total = sum(n.distance_to_completion for n in nodes)
    return [costs.get(id(n), INF) + total - n.distance_to_completion for n in nodes]


def _repetition_cost(node: Repetition, costs: dict[int, Cost]) -> Cost:
    if node.max < 1:
        return INF
    inner = node.node
    extra = max(node.min - 1, 0) * inner.distance_to_completion
    return 1 + costs.get(id(inner), INF) + extra


def reach_costs(
    grammar: "fandango.language.grammar.grammar.Grammar", target: Node
) -> dict[int, Cost]:
    """Map id(node) to the size of the smallest subtree from node that
    contains `target`. Nodes that cannot reach it are absent or infinite."""
    nodes = _all_nodes(grammar)
    if not any(n is target for n in nodes):
        raise FandangoValueError(
            f"{target.format_as_spec()} is not a node of this grammar "
            "(targets are matched by identity)"
        )
    costs: dict[int, Cost] = {id(target): target.distance_to_completion}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node is target:
                continue
            if isinstance(node, NonTerminalNode):
                if node.symbol in grammar.generators:
                    continue
                cost = 1 + costs.get(id(grammar.rules[node.symbol]), INF)
            elif isinstance(node, Alternative):
                cost = 1 + min(costs.get(id(a), INF) for a in node.alternatives)
            elif isinstance(node, Concatenation):
                cost = 1 + min(_carrier_costs(node.nodes, costs), default=INF)
            elif isinstance(node, Repetition):
                cost = _repetition_cost(node, costs)
            else:
                continue
            if cost < costs.get(id(node), INF):
                costs[id(node)] = cost
                changed = True
    return costs


def _choose(options: list[tuple[int, Cost]], budget: Cost, shortest: bool) -> int:
    """Pick an index from (index, cost) pairs with finite cost."""
    best = min(c for _, c in options)
    if not shortest:
        fitting = [i for i, c in options if c <= budget]
        if fitting:
            return random.choice(fitting)
    return random.choice([i for i, c in options if c == best])


class _Walker:
    def __init__(
        self,
        grammar: "fandango.language.grammar.grammar.Grammar",
        target: Node,
        costs: dict[int, Cost],
        shortest: bool,
    ):
        self.grammar = grammar
        self.target = target
        self.costs = costs
        self.shortest = shortest

    def cost(self, node: Node) -> Cost:
        return self.costs.get(id(node), INF)

    def walk(
        self, node: Node, parent: DerivationTree, budget: int, in_message: bool
    ) -> None:
        if node is self.target:
            node.fuzz(parent, self.grammar, max(budget, 0), in_message)
        elif isinstance(node, NonTerminalNode):
            self._nonterminal(node, parent, budget, in_message)
        elif isinstance(node, Alternative):
            options = [
                (i, self.cost(a))
                for i, a in enumerate(node.alternatives)
                if self.cost(a) < INF
            ]
            i = _choose(options, budget, self.shortest)
            self.walk(node.alternatives[i], parent, budget, in_message)
        elif isinstance(node, Concatenation):
            self._concatenation(node, parent, budget, in_message)
        elif isinstance(node, Repetition):
            self._repetition(node, parent, budget, in_message)
        else:
            raise FandangoValueError(
                f"Cannot reach the target through {node.format_as_spec()}"
            )

    def _nonterminal(
        self,
        node: NonTerminalNode,
        parent: DerivationTree,
        budget: int,
        in_message: bool,
    ) -> None:
        sender = recipient = None
        if not in_message and node.sender is not None:
            sender, recipient = node.sender, node.recipient
            in_message = True
        tree = DerivationTree(
            node.symbol, [], sender=sender, recipient=recipient, read_only=False
        )
        parent.add_child(tree)
        self.walk(self.grammar.rules[node.symbol], tree, budget - 1, in_message)

    def _fill(
        self,
        node: Node,
        parent: DerivationTree,
        budget: Cost,
        in_message: bool,
    ) -> None:
        """Fuzz an off-path position as Concatenation.fuzz does."""
        if node.distance_to_completion >= budget:
            node.fuzz(parent, self.grammar, 0, in_message)
        else:
            node.fuzz(parent, self.grammar, int(budget), in_message)

    def _concatenation(
        self,
        node: Concatenation,
        parent: DerivationTree,
        budget: int,
        in_message: bool,
    ) -> None:
        carrier_costs = _carrier_costs(node.nodes, self.costs)
        options = [(i, c) for i, c in enumerate(carrier_costs) if c < INF]
        carrier = _choose(options, budget, self.shortest)
        # What each position still has to leave for the positions after it.
        needs = [
            self.cost(n) if i == carrier else n.distance_to_completion
            for i, n in enumerate(node.nodes)
        ]
        remaining = float(budget)
        for i, child in enumerate(node.nodes):
            before = parent.size()
            share = remaining - sum(needs[i + 1 :])
            if i == carrier:
                self.walk(child, parent, int(share), in_message)
            else:
                self._fill(child, parent, share, in_message)
            remaining -= parent.size() - before

    def _repetition(
        self,
        node: Repetition,
        parent: DerivationTree,
        budget: int,
        in_message: bool,
    ) -> None:
        inner = node.node
        low = max(node.min, 1)
        reps = low
        if not self.shortest:
            # Only counts whose minimal completion fits the budget, as the
            # normal fuzzer stops repeating when the budget runs out.
            spare = budget - self.cost(inner) - (low - 1) * inner.distance_to_completion
            fit = low + int(max(spare, 0) // max(inner.distance_to_completion, 1))
            reps = random.randint(low, max(low, min(node.max, fit)))
        carrier = random.randrange(reps)
        node.iteration += 1
        iteration = node.iteration
        remaining = float(budget)
        start = len(parent.children)
        for rep in range(reps):
            later = max(node.min - rep - 1, 0) * inner.distance_to_completion
            if rep < carrier:
                later += self.cost(inner)
            share = remaining - later
            before = parent.size()
            if rep == carrier:
                self.walk(inner, parent, int(share), in_message)
            elif (
                inner.distance_to_completion >= share
                and rep >= node.min
                and rep > carrier
            ):
                break
            else:
                self._fill(inner, parent, share, in_message)
            for child in parent.children[start:]:
                child.origin_repetitions.insert(0, (node.id, iteration, rep))
            start = len(parent.children)
            remaining -= parent.size() - before


def fuzz_through(
    grammar: "fandango.language.grammar.grammar.Grammar",
    require: Requirement,
    start: Union[str, NonTerminal] = "<start>",
    max_nodes: int = 50,
    shortest: bool = True,
    costs: Optional[dict[int, Cost]] = None,
) -> DerivationTree:
    """A random tree from `start` whose derivation uses `require`.

    `require` is `(rule, alt_index)` for a top-level alternative of `rule`, or
    any node of the grammar (see `alternative_branches`). Raises
    `FandangoValueError` if the target cannot be reached from `start`.
    """
    target = resolve(grammar, require)
    start_nt = _nt(start)
    if costs is None:
        costs = reach_costs(grammar, target)
    entry = NonTerminalNode(start_nt, grammar.grammar_settings)
    if (
        start_nt in grammar.generators
        or costs.get(id(grammar.rules[start_nt]), INF) == INF
    ):
        raise FandangoValueError(
            f"{target.format_as_spec()} cannot be reached from "
            f"{start_nt.format_as_spec()}"
        )
    root = DerivationTree(start_nt)
    _Walker(grammar, target, costs, shortest).walk(entry, root, max_nodes, False)
    tree = root.children[0]
    tree._parent = None
    return tree


def fuzz_alternatives(
    grammar: "fandango.language.grammar.grammar.Grammar",
    rules: Iterable[Union[str, NonTerminal]],
    k: int,
    start: Union[str, NonTerminal] = "<start>",
    max_nodes: int = 50,
    shortest: bool = True,
    nested: bool = False,
) -> dict[tuple[NonTerminal, int], list[DerivationTree]]:
    """`k` trees per alternative of each rule in `rules`.

    Keys are `(rule, index)`, with index into `alternatives(grammar, rule)`,
    or into `alternative_branches(grammar, rule)` when `nested=True`. An
    alternative that cannot be reached from `start` maps to an empty list.
    """
    out: dict[tuple[NonTerminal, int], list[DerivationTree]] = {}
    for rule in rules:
        nt = _nt(rule)
        branches = (
            alternative_branches(grammar, nt) if nested else alternatives(grammar, nt)
        )
        for index, branch in enumerate(branches):
            costs = reach_costs(grammar, branch)
            try:
                out[(nt, index)] = [
                    fuzz_through(grammar, branch, start, max_nodes, shortest, costs)
                    for _ in range(k)
                ]
            except FandangoValueError:
                out[(nt, index)] = []
    return out
