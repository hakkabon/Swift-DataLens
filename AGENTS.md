# Working agreements for this repo

Local-regression toolkit (Cleveland-style LOESS first, then clean-room
Loader-style adaptive smoothing / local likelihood). AGENTS.md duplicated
from sibling conventions, trimmed to this scope.

## Commands

```bash
swift build                  # library
swift test                   # full suite (fast scaffold; grows with the LOESS port)
swift test --filter DataLensTests
swift run Benchmarks         # micro-benchmarks (debug numbers; compare relatively)
```

`swiftformat` / `swiftlint` configs exist at repo root; run before submitting.

## Code conventions

- **Value semantics**: models are `struct`s conforming to `Sendable`. No
  classes for numerics.
- **Seeded RNGs everywhere**: every sampler offers `random()` (system RNG)
  and `random(using: &rng)` for any `RandomNumberGenerator`
  (`SeedableRandomNumberGenerator(seed:)` in tests). Never use `arc4random`.
- **Validation via `precondition`** with a message stating the requirement.
  Return `nil` only for data-dependent failures (empty input, singular
  design, non-convergence); never trap on representable edge cases —
  guard `Int()` conversions from `Double` (e.g. `floor(±inf)`).
- **No return-type-only overloads**: use distinct names
  (`randomDouble(in:)` / `randomInt(in:)`).
- **Linear algebra**: least-squares fits go through the vendored QR
  least-squares seam — never form XᵀX for fitting. A small square solver
  is only for Newton-type systems and leverage/SE weights.
- **Iterative fits** (robustness reweighting, bandwidth selection, IRLS/EM
  if local likelihood lands) must have: backtracking or step-halving, an
  iteration cap, and where feasible a score-at-solution check that fails
  loudly instead of returning unconverged values.
- **Public API is documented** with doc comments; keep DocC
  (`Sources/DataLens/DataLens.docc/`) symbol lists in sync when adding types.
- Prefer pure Swift + Foundation. No new FFI surfaces without explicit
  approval. No GPL code: Loader-style work is clean-room (never `locfit`).

## Test conventions

- **Deterministic**: fixed seeds (`SeedableRandomNumberGenerator(seed:)`),
  exact assertions at 1e-9 where closed forms exist.
- Statistical tests use fixed seeds with generous tolerances, never
  million-sample marathons.
- New fits need: parameter recovery on synthetic truth, plus an
  invariant (residual/gradient check, deviance < null, likelihood
  improvement).
- Keep the debug suite fast: shrink n/iterations before widening
  tolerances. If a single test exceeds ~5s in debug, reduce it.

## Docs

- Update `README.md` (features, API table, structure tree, test count,
  limitations) and DocC topics in the same change as the feature.
- `docs/DECISIONS.md` records *why* for traps and judgment calls — add an
  entry when debugging uncovers something the next person would trip over.
