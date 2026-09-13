# Decision log

One paragraph per decision that future work depends on. Newest last.

## 1. LOESS gets its own repo (DataLens)

`Loess.swift` in Numerical-Statistics touches only `LinAlg` +
`Descriptive` (+ distributions in tests), so extraction is mechanical.
This repo incubates local regression: Cleveland-style LOESS first
(Cleveland 1979 / Cleveland–Devlin 1988), then clean-room Loader-style
adaptive smoothing and local likelihood. Keep KDE in the old repo; move
Loess, its tests, and its benches here. Never `locfit` (Loader's GPL
package) — no GPL code enters this repo.

## 2. Module name `DataLens`

Repo `Swift-Data-Lens`, package `Swift-Data-Lens`, library/target
`DataLens` (`import DataLens`), tests `DataLensTests`. Rationale: the
default `Swift-Data-Lens` target builds as module `Swift_Data_Lens` and
the scaffolded test import was already broken. Normalized while the repo
was fresh (3 commits) — second and last breaking rename.

## 3. Vendor minimal numerics, no dependency on Numerical-Statistics

`Loess.swift` needs `LinAlg.leastSquares` (Householder QR),
`Regression.solve` (square Newton/leverage systems), and
`Descriptive.median` (robustness scale). Vendoring a minimal,
self-contained subset keeps DataLens dependency-free and matches the
old repo's "self-contained by design" seam (`nearestIndices` stays the
kd-tree drop-in point). Full `LinAlg`/`Regression`/`Descriptive` stay in
the old repo.

## 4. Scope order: classic LOESS, then adaptive

Port exact-reproduction behavior first (linear/plane/quadratic
reproduction, bisquare robustness, SEs, smoother trace, GCV span
selection) with the existing `LoessTests` as the gate. Only then add
Loader-style adaptive bandwidths and local-likelihood families as new,
clean-room types — no behavior change to classic `Loess` without a
DECISIONS entry.

## 5. LOESS port record (byte-identical engine, internal numerics)

`Sources/DataLens/Loess.swift` is a byte-identical copy of
Numerical-Statistics' `Loess.swift` (verified with `diff`); only the
helpers moved, demoted from `public` to internal under
`Sources/DataLens/Internal/`: full `LinAlg.swift` (QR + least squares;
`qr(_:)` kept for seam completeness), `Regression.solve` and
`Descriptive.median` as single-function subsets, and `SeededRNG.swift`
(test-only — `Benchmarks` uses deterministic sine data so the executable
needs no RNG). `LoessTests` was converted XCTest → swift-testing with
1:1 assertions, same seeds (2201/2202/2203) and tolerances, minus the two
`_ = rng` no-ops. The robustness 2-cycle fixes came along verbatim
(bisquare scale floor at 1e-8 × y-MAD; degenerate neighborhoods fall back
to a robustness-ignoring local mean, never a single neighbor). Gate: 10
tests green in ≈1s debug. Future adaptive work must not change classic
`Loess` behavior without an entry here, and any divergence between the
vendored numerics and their old-repo originals needs one too.

## 6. kd-tree integration (adapted N-D port, routed paths)

The tree was provided as a chat-attached 2D (`Point2D`) file and vendored
as `Internal/KDTree.swift`, generalized 2D → N-D (`axis = depth % dims`)
— LOESS is multivariate, so the 2D original could not serve. Two parity
fixes were required for exact brute-force agreement (proven by
`NearestNeighborTests` over seeded clouds with duplicates, coincident
queries, and degenerate inputs): the result queue orders lexicographically
by (distance, index) instead of distance-with-traversal-order ties, and
the backtrack condition is `<=` (strict `<` misses equidistant
smaller-index points across the split plane). Queue insertion is O(k)
ordered-insert, not sort-per-insert. Path selection, all behind the
internal `NeighborSearch` (built once per `fit`, since a per-query rebuild
loses to brute force): tree iff n ≥ 64 and k·4 ≤ n — filling the queue
already costs O(n·k), so near-n neighborhoods visit the whole tree and
lose to one brute-force sort; single-shot `predict`/`standardError` skip
the build (`forBatchUse: false`), which is optimal for one query.
`Loess.swift` internals were threaded (`localFit` takes the search) with
no public API change and no behavior change — the pre-tree suite passes
unaltered. The 64 / 4× constants are debug-bench heuristics; recalibrate
in release mode if spans drift. `DataLens.version` stays 0.1.0.

## 7. Solver seam bound to NumericCore (LAPACK on Apple, vendored fallback)

`LinAlg.leastSquares` and `Regression.solve` now call
`NumericCoreAccelerate.AccelerateBackend` (thin QR via dgeqrf/dormqr/dtrtrs
at e884295) on Apple platforms, keeping the exact signatures and the
nil-on-rank-deficiency contract — `Matrix(rows:)` converts row-major to
column-major at the boundary and throws (→ nil) on empty/ragged input,
matching the old guards. Two known divergences, both benign here: the
rank check is an absolute 1e-12 on R's diagonal vs the old scaled
1e-12·max(|R₀₀|,1), and `solve([], [])` is now nil instead of `[]`
(strictly safer — the old `[]` would have trapped at the `col[0]` call
sites, which are unreachable anyway). Linux is guarded by
`#if canImport(Accelerate)` plus a platform-conditional product dependency,
falling back to the vendored Householder/elimination bodies — which is why
those bodies stay despite the binding. `qr(_:)` has no Accelerate
counterpart and stays vendored everywhere. Gate: full suite (now 18 with
`SolverSeamTests` pinning the nil contract and closed-form solves) green
on the LAPACK path, and debug benches improved (fit 2235→1773ms, predict
1215→807ms). One compiler note: `guard let v = try?` flattens the
double-optional, so the bodies use `v.storage`, not `v?.storage`.

## 8. SPD seam bound, LOESS calls untouched

`Regression.solveSPD` binds `AccelerateBackend.solveSPD` (dpotrf/dpotrs)
on Apple, elimination in the Linux fallback — same signature/contract
shape as `solve`, nil iff not positive-definite. Deliberately *not*
rerouted into the frozen LOESS leverage/SE calls (those XtX matrices are
SPD-ish but only up to 1-ulp asymmetry from accumulation order, and
`dpotrf` reads one triangle): classic `Loess` stays on the QR `solve`
per entry 5, and `solveSPD` waits for its designed consumer — the local-
likelihood normal equations, SPD by construction. Symmetry is *not*
checked (LAPACK can't); the doc comment carries the warning instead.
Shared elimination extracted to `eliminate(_:_:)` serving both fallbacks.
Gate: 20 tests (new `solveSPD` closed-form + indefinite/zero/non-square
nil cases) green on the Cholesky path, which also retires that file's
unverified-marshaling caveat for these entry points.

## 9. NumericCore pinned by revision, not version

`from: "0.1.0"` fails to resolve with `Revision c9892c7 ... version 0.1.0
does not match previously recorded value 6d45de7` — even though the remote
advertises the tag correctly (`ls-remote`: 0.1.0 → c9892c7) and every
fetch mirror on disk is correct. Proven pre-existing-data-independent via
a scratch project, and surviving `package reset` plus global-cache clears,
so it is not local staleness; likely a stale tag mapping on the fetch
path (the tag may first have pointed at 6d45de7) or an SPM quirk with the
annotated tag. Workaround: pin `revision: "c9892c7..."` (the tagged
commit itself) — resolution, build, and all 20 tests pass. This locks
exactly the tested code, which is what a release commit wants anyway;
bump manually on new NumericCore tags (`swift package update` won't move a
revision pin). Retry `from:` if the tag is ever deleted and re-pushed.
