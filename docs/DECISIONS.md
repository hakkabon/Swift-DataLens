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

## 10. Adaptive smoothing via per-observation local AICc (0.2.0)

`AdaptiveLoess` scores candidate neighborhood sizes at each fit point and
keeps the minimizer — a clean-room adaptive design (no `locfit` code),
not a port. Three judgment calls, all load-bearing. **(1) Per-observation,
not totals.** Totals (`k·ln(RSS/k) + …`) let the neighborhood SIZE dominate
instead of fit quality: the `k·ln(RSS/k)` term grows with k whenever
RSS/k > 1, so at a gross outlier it selected k=4, the robustness rounds
then zeroed the whole contaminated cluster, and the fit trapped far from
truth (caught by the ported outlier test). Dividing by k keeps the
bias–variance reading intact — large-k dilution then wins at outliers and
recovery works. **(2) Select-once-then-reweight.** Selection runs once,
unweighted (mirroring how `Loess` fixes its span across rounds); the
bisquare rounds refine on fixed neighborhoods. Selection stays a pure
function of the data. **(3) Shared engine, frozen behavior.** `Loess`
gained no behavior: `localFit` and `standardError` delegate their QR +
leverage / kernel-norm blocks to `LocalPolynomial.fitWeighted` and
`Loess.kernelStandardError` (same ops, same order — the 20-test suite
passes unaltered), and `AdaptiveLoess` reuses those plus `NeighborSearch`
and the fallback cascade (including the robust-ignoring bounded mean for
degenerate combined weights). Candidates default to a ~1.6× geometric grid
(≤9 values, always ending at n so flat regions may go global); exact ties
resolve toward larger k; `k ≥ q+2` required, else nil. Evidence on
heterogeneous truth (flat left, growing oscillations right): mean selected
k 15.4 flat vs 7.1 wiggly, RMSE 0.038 vs best fixed span 0.248; on
homogeneous sine, 0.047 vs 0.061. `DataLens.version` → 0.2.0.

## 11. Local likelihood via Newton–IRLS (0.3.0)

`LocalLikelihood` (Gaussian/Binomial/Poisson) maximizes the
locality-weighted log-likelihood per fit point with Newton–IRLS:
step-halving on the weighted local deviance, 25-iteration cap, 1e-8
relative tolerance, η clamped at ±30 for weight arithmetic. The normal
equations go through `Regression.solveSPD` — the consumer that seam was
added for. Three calls worth recording. **(1) No robustness rounds.**
Likelihood families carry their own variance structure; bisquare
reweighting rounds are future work, not this release. **(2) Boundary MLEs
saturate, never diverge.** Separated neighborhoods (all-0/all-1) clamp
finitely (measured 2e-10/1−2e-10 on step truth); rank-deficient systems
fall back to a bounded locality mean mirroring `Loess`. Silent fallback
(not loud failure) is deliberate and consistent with `Loess`: what returns
is never an unconverged iterate. **(3) Gaussian consistency as gate.**
Without robustness, Gaussian local likelihood IS the Loess WLS estimator —
but `Loess` always applies one bisquare round (even `robustIterations: 0`
updates the weights once), so cross-agreement is asserted loosely (0.05)
while exactness is pinned directly: normal-equations residual ≤1e-9,
binomial score ≤1e-6. Measured recovery: binomial RMSE 0.060, Poisson
0.44, deviances ~2× below null in both. `DataLens.version` → 0.3.0.

## 12. Batch evaluation + concurrency (0.4.0)

Batch overloads (`predict(_:)` / `standardErrors(at:)` taking arrays) on
all three smoothers share one `NeighborSearch`, while single-point calls
keep skipping the tree build — the per-call rebuild was pure waste for
grids. `kernelStandardError` now takes the search (internal signature
change only). Concurrency is indexed task groups over a `concurrentMap`
helper: per-point arithmetic is unchanged, reductions (medians, traces,
deviances) stay sequential in index order, so concurrent results are
bit-identical to sequential ones — asserted with `==`, not tolerance, in
`BatchTests`. Two deliberate namings: `*Concurrently` suffixes instead of
async overloads (sync/async overload ambiguity at call sites), and
robustness rounds stay sequential across rounds (each needs the previous
residuals) with parallel inner loops — the compiler itself enforces the
per-round snapshots by rejecting captured `var`s, which documents the
semantics. `selectSpan` stays sync (spans are whole fits; parallelizing
that loop is future work). `DataLens.version` → 0.4.0.

## 13. Automated tuning in one call (0.5.0)

`AutomaticSmoother` routes by response type — all-0/1 → binomial, integral
non-negative → Poisson, else continuous — then tunes by a shared
criterion: AIC for the likelihood legs, GCV for continuous, where
`AdaptiveLoess` and fixed-span `Loess` compete head-to-head (GCV is
computable from fitted values + trace for both, so the comparison is
apples-to-apples). Three notes. **(1) Integral means counts, exactly.**
`y == y.rounded()` with no tolerance: true counts are exact in Double,
and tolerance would misroute near-integers; binary is checked first since
0/1 are also integral. **(2) Fallbacks are reported, not hidden.** Tiny
inputs with no valid adaptive neighborhood, or a failed likelihood path,
degrade to the next option with a note in `TuningSummary` — verified by a
test that asserts on the note's content, since silent rerouting would be
worse than failing. **(3) The tuner returns fits, not just predictions.**
`FittedSmoother` carries the full underlying fit (predict/batch/SE all
forward), so nothing downstream is lost by going through automation.
Measured: heterogeneous truth routes adaptive and beats every fixed span;
binary/counts recoveries hold at the likelihood tests' margins.
`DataLens.version` → 0.5.0.

## 14. Predictive flexibility: gradients, extrapolation, missing data (0.6.0)

Three additions, all default-preserving (existing calls behave exactly as
before). **(1) Gradients** are first-order local coefficients (exact on
lines/planes; sine tracks cosine within 0.15); degree-0 and degenerate
fits yield nil, honestly. **(2) Extrapolation is a query-time policy**
(`.polynomial` default = status quo, `.nearest` = edge fitted value,
`.unavailable` = NaN/nil), never a refit — the fit is policy-independent,
so switching is free. The hull is a per-dimension box, documented as such;
`.nearest` SEs come from the edge point, consistently with the value.
Gradients deliberately ignore the policy (local-polynomial objects).
**(3) Missing data drops rows** (`droppingMissing`, off by default) with
`keptIndices` reporting survivors for join-back; width mismatches still
fail loudly (schema errors are not missing data), and the count-mismatch
guard runs before dropping so truncation can never hide it. Appended-NaN
fits are bit-identical to clean fits. `DataLens.version` → 0.6.0.

## 15. Linux skips NumericCore entirely (CI red → green)

The Linux CI job failed without compiling a line of DataLens: the
`NumericCoreFFI.xcframework` ships ios-arm64, ios-simulator, and macOS
slices but no Linux slice, so `NCBindings` (which `NumericCore` depends
on, which we import) fails with ~100 missing-symbol errors
(`RustBuffer`, `uniffi_*`, …). That is an upstream packaging bug — the
fix belongs in Swift-NumericCore (Linux slice, or conditional FFI with
the pure-Swift fallback it already has). The DataLens-side hardening,
done here: both NumericCore products are now Apple-platform-conditional
in `Package.swift`, and `LinAlg`/`Regression` gate on
`canImport(NumericCoreAccelerate)` (imports and code paths together) —
Linux builds link zero NumericCore modules and run the vendored
Householder/elimination bodies, which is exactly what those bodies were
kept for. macOS still binds LAPACK (both modules present in the build,
62 green, bench signature unchanged). No version bump: behavior is
identical on every platform that built before.

## 16. Fallback Cholesky restores solveSPD parity on Linux

CI proved the `solveSPD` fallback wrong, not just slow: Gaussian
elimination solves nonsingular indefinite systems (returning `[1, 1]`
for `[[1, 2], [2, 1]]`), while LAPACK's `dpotrf` returns nil — same
contract name, different verdicts, one red Linux test. The fallback is
now scalar Cholesky from the upper triangle (mirroring `uplo = "U"`),
nil on a non-positive pivot exactly like `info > 0`; `eliminate` stays
for general `solve`, which must keep solving indefinite systems.
Verified three ways: direct `cholesky` unit tests (pinned on every
platform), the full 63-test suite under a temporarily forced fallback
(`#if false`, reverted — the exact code Linux executes), and the
restored Accelerate path afterward. Lesson: fallback paths need
verdict-parity tests, not just value tests — the next fallback addition
should ship its indefinite/singular cases with it.
