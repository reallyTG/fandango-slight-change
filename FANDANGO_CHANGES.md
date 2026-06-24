# Fandango parser changes (for upstream PR)

This fork (`reallyTG/fandango-slight-change`) carries targeted fixes to the
Earley parser in `src/fandango/language/grammar/parser/`. They are independent,
upstreamable bug fixes — this document tracks them so we can prepare a clean PR
to `fandango-fuzzer/fandango`.

Both were found while parsing real-world inferred/reference grammars (grammar
inference tools — Arvada/TreeVada/Stalagmite — ported to `.fan`), which exercise
ambiguity and nullable recursion far harder than the hand-written `.fan` specs in
the test suite.

---

## 1. ParseState hash/eq inconsistency defeated Earley dedup (perf)

**Commit:** `fadedb1f`
**File:** `parser/parse_state.py`

`ParseState.__hash__` included `tuple(self.children)` while `__eq__` compared only
`(nonterminal, position, symbols, dot)`. This violates the Python hash contract
(equal objects must hash equal). `Column` dedups states with a `set`, which buckets
by hash before calling `__eq__`, so two `__eq__`-equal states carrying different
derivation subtrees landed in different buckets and were both kept.

On an ambiguous grammar the chart then accumulated one state per derivation **tree**
instead of one per Earley **item**, making recognition exponential — a 10-element
JSON array and small tinyc programs timed out at >30s, dominated by ~1M
`copy.deepcopy` of partial trees in `complete()`.

**Fix:** key `__hash__` on `(nonterminal, position, symbols, dot)` — the standard,
complete Earley item identity, now consistent with `__eq__`. Parsing becomes
polynomial on ambiguous grammars (8-element JSON array 4652ms → 15ms). The only
behavioural change is that one parse tree per item is retained rather than all of
them (recognition and first-tree extraction are unaffected).

---

## 2. Nullable-completion ordering bug silently mis-recognised valid input (correctness)

**File:** `parser/iterative_parser.py` (`predict`) + `parser/column.py`
(`complete_map` / `find_complete`)

**Symptom:** the parser **silently rejected valid strings** (returns no tree, not an
error) for grammars using the standard fuzzingbook list idiom:

```
<elements> ::= <element> <tail>
<tail>     ::= '' | <sep> <tail>
<sep>      ::= ',' <elements>
```

A single element parses; any comma-separated list of two or more (`[1,2]`,
`[true,false]`) was rejected. This idiom appears in essentially every list/
separator construct of the Stalagmite and Mimid reference and inferred grammars,
so it broke the inference-repair oracle wholesale.

**Root cause:** the classic Earley nullable-completion ordering problem
(Aycock & Horspool 2002). The engine processes each column in a single pass. When a
nullable nonterminal `X` has a **zero-width finished item** `X → γ•` at `(k, k)` that
the completer processes *before* a waiter `A → α • X β` is added to column `k`
(here the waiter arrives only when `<sep>` finishes, late in the column), the
completer for the nullable `X` has already run and is never revisited. When the
waiter is later reached it only *predicts* `X` (re-adding already-deduped rules) and
is never advanced over the empty `X`, so the completion chain to the start symbol
never closes.

**Fix:** after `predict` adds a symbol's rules, re-run `complete()` for any
zero-width (`position == k`) finished state of that symbol already present in the
column. This advances the just-added waiter and is a no-op (deduped) for waiters
already advanced — so it is correct regardless of the order in which the finished
item and the waiter appear. To keep it O(1) rather than scanning the column,
`Column` now indexes finished states by nonterminal in a `complete_map`
(maintained in `add`/`replace`, mirroring the existing `dot_map`), exposed via
`find_complete()`.

**Verification:** minimal repro and the full Stalagmite golden JSON grammar now
accept all valid lists and still reject malformed ones (`[1,]`, `[1,,2]`, `[1 2]`).
Full Fandango suite: 847 passed, 0 new failures (pre-existing failures are
environmental — missing `faker` module and C/clang toolchain). Downstream repair
suite: 61 passed.
