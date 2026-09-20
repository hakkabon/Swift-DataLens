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

## 17. Cooperative cancellation on the concurrent surface (0.6.2)

All 15 concurrent entry points (`*Concurrently`, `fitConcurrently`)
are now `async throws`: each task checks cancellation before starting
its point, so cancelling aborts pending points with `CancellationError`
while in-flight points finish (CPU work cannot be preempted). The
`throws` is source-breaking in principle but affects only the month-old
async API with no downstream callers yet. Tests pin both sides:
cancellation-aborts (1000 trivial tasks, immediate cancel — deterministic
by oversubscription, no timing involved) and uncancelled-completes. Note
for the app track: scroll-driven grids should cancel superseded fits
rather than letting them pile up behind a pinching finger.

## 18. Borrowed-bandwidth fast paths for adaptive grids

App-track profiling showed the adaptive smoother dominating interactive
cost twice: full auto-tune on 1000 sine rows (~38s release) and, worse,
every 200-point grid re-running per-point AICc selection (~16s release
for means + SEs). Reading `predictAt`/`standardErrorAt` gave the cost
model: per query, ~C candidates × (nearest + 2 QR fits) plus a
`BoundingBox` rebuilt from scratch each query.

The fast paths (`predictFast`, `standardErrorsFast`, each in
single/batch/concurrent) reuse the nearest training point's already
selected neighborhood and run the identical robust local evaluation at
the query. On dense smooth data the borrowed bandwidth selects (nearly)
what a fresh selection would; the release bench says ~25× on means and
~45× on SEs (n=60, x200 grid), with agreement inside half the noise
scale (pinned: 0.039/0.037 at seed 4245). Degenerate inputs degrade
through the same fallback cascade, agreeing to solver noise.

Deliberate boundaries: exact `predict`/`standardErrors` are untouched
and remain the default (tests, publications); the fast variants are
opt-in and documented as approximations. The per-query hull-box rebuild
was hoisted in the existing batch paths too — pure motion, no numeric
change, proven by the untouched 65-test suite going 68 green with the 3
new agreement tests.

## 19. Opt-in shallow tuning via `adaptiveContender`

App-track profiling left the adaptive *fit* as the wall after fast grids
landed: full auto-tune on 1000 sine rows runs ~44s release, of which the
adaptive contender is only ~10s at robust-4 (robust refits dominate) —
but inside the interactive budget (1 span, 1 robust round) it is ~9s of
~14s. So `AutomaticSmoother.fit` gains `adaptiveContender: Bool = true`:
`false` skips the adaptive leg, tunes fixed-span Loess only, and records
the skip in `TuningSummary.notes` (the Loess reason string changes too,
so no summary ever claims a competition that didn't happen). Routing is
untouched — binary/counts still try likelihood first — and the default
preserves exact prior behavior. Measured: 14.1s → 5.2s release on the
interactive budget; 43.7s → 33.9s on full defaults. The flag trims a
contender, never quality silently: shallow fits say they are shallow.

## 20. `selectSpan` scored pre-drop rows (latent nil on missing data)

The shallow flag exposed it: `Loess.selectSpan` zipped pre-drop `trainY`
against post-drop `fittedValues`. Any dropped row misaligned every pair
and NaN-poisoned rss, so the score never beat infinity and selection
returned nil on ALL missing-data input. It never surfaced because the
adaptive contender usually won and the nil Loess leg was silently
ignored. Fixed by scoring on the fit's own dropped arrays; pinned by a
regression test proven to fail without the fix. Audit found no sibling:
every other rss/deviance zip in `Loess`, `AdaptiveLoess`, and
`LocalLikelihood` operates on post-drop locals inside fit bodies.
Lesson: cross-boundary zips (caller arrays × fitted arrays) are the
shape to grep for whenever a new dropping path is added.

## 21. Nadaraya–Watson as a thin wrapper over degree-0 kernels

App-track feature request (old-app parity: kernel regression alongside
LOESS). Implemented as `NadarayaWatson`, deliberately thin: every local
evaluation delegates to `Loess.localFit` with degree 0 (identical values,
fallback cascade, and leverage) and SEs go through
`Loess.kernelStandardError` the same way — no duplicated math to drift.
What the type owns: span-fraction neighborhoods (so a tuner can compare
it against `Loess` directly), the trace/sigma bookkeeping, span
selection scored on dropped rows (the #20 rule), and analytic tricube
gradients including bandwidth variation (omitting dh/dx errs by O(1);
the span neighborhood makes the mean piecewise smooth, so gradient
tests query off-lattice points and the docs state the branch rule).
Pinned by bit-parity with `Loess.fit(degree: 0)` — if the kernels ever
diverge, that test names the commit. Not auto-routed in
`AutomaticSmoother` (that would silently move every tuned fit); explicit
selection belongs to the app track.

## 22. `FittedSmoother` carries kernel fits without routing them

App track needs `NadarayaWatson` behind the uniform evaluation seam
(`ChartModel` predicts through `FittedSmoother`), while the tuner must
never select it (explicit-selection decision). So the enum gains a
`nadarayaWatson` case with all 11 pass-throughs, and the two exhaustive
switches name it: the fallback strings it defensively, and the routing
test now pins that continuous data never arrives as kernel. Round-trip
test proves wrapped == direct on every path. Rule going forward: new
smoothers get a carrier case on arrival, routing only by separate
decision.

## 23. Whittaker–Eilers over a local band solver (Takahashi included)

App-track feature request (penalized smoothing alongside LOESS/kernel).
`ŷ = (I + λDᵀD)⁻¹y` is SPD-banded, so the fit goes through a new
internal `BandedMatrix` (unit-Cholesky + Takahashi band-inverse) rather
than the solve-only `LinAlg` seam, which cannot yield the trace or the
inverse band the GCV criterion and standard errors require. The solver
is proven element-wise against dense references (factor, solve, and
inverse band to 1e-9; nil-verdict parity on non-SPD per #16), including
two bugs the tests caught before review: a closed-range back-substitution
overrun and the classic LLᵀ-vs-LDLᵀ Takahashi scaling (diagonal scale
applies to δ only — verified by hand on 2×2 first). Exact SEs come from
full solved rows (O(n²) fit-time, documented ~10k-row comfort zone);
grids interpolate fitted values and SEs alike. Missing rows drop with
`keptIndices` like every other smoother (in-system weight-0 is a
documented follow-up); non-uniform spacing is documented, not solved
(x orders, spacing ignored). Order-2 + chosen λ *is* the HP filter —
no separate API blesses a λ convention this repo cannot defend on
arbitrary x. Not auto-routed; explicit selection only, same as kernel.

## 24. Total variation via ADMM, verified by KKT

App-track request (edge-preserving complement to the smooth family).
Implemented as 1-D fused lasso through ADMM — not the faster Condat
direct algorithm — deliberately: ADMM is correct by construction
(textbook Boyd splits, banded direct x-step through `BandedMatrix`),
while a from-memory Condat risks silent wrongness that even good tests
might shape around. Instead the tests verify OPTIMALITY directly: the
KKT system `w = λDᵀs` makes duals a forward recurrence
(`s[0] = −w[0]/λ`, `s[i] = s[i−1] − w[i]/λ`), so bound violations, the
closing equation, and jump-sign agreement are all exactly checkable
with no reference implementation — any solver wrongness fails loudly
there. Iterative per house rules (residual stopping test, 20k cap, nil
past it). Two more deliberate approximations, both documented on the
type: homoskedastic σ bands (a piecewise-constant fit has no meaningful
pointwise leverage) and segment-count trace for GCV (Tibshirani–Taylor,
estimated under a relative jump tolerance). Not auto-routed; carrier
case only, same as kernel and Whittaker.

## 25. Additive main effects use centered LOESS backfitting

Phase 3 needs interpretable multi-predictor structure without turning the
multivariate LOESS surface into a black box. `AdditiveModel` therefore fits
`y = α + Σfⱼ(xⱼ)` by cyclic backfitting, reusing the frozen one-dimensional
`Loess` implementation for every term. Each update is centered over the
training rows; this makes α the response mean and prevents arbitrary constants
from migrating between terms. A term specification owns predictor index, span,
and degree, so selected-variable and heterogeneous-smoothness models need no
second API. The first layer is deliberately Gaussian and main-effects-only:
likelihood families, categorical effects, interactions, and joint covariance
need explicit statistical contracts rather than silent approximations.

The solver follows the repository's fail-closed iterative rule: convergence is
maximum pointwise component change relative to response scale, capped at 100
cycles by default, and an exhausted fit returns nil. Robust LOESS rounds are
available but default off because linear smoothers give the clearest classical
backfitting behavior. Effective degrees of freedom are documented as the
intercept plus centered component traces; sigma is correspondingly approximate,
not a joint covariance claim. Deterministic orthogonal-grid tests pin recovery,
centering, decomposition, local-polynomial derivative recovery, whole-row
missing-data handling, selected terms, validation, and the non-convergence verdict.

## 26. Mature GAMs through the unified specification, not a second workflow

The additive foundation was useful only through its direct `AdditiveModel`
entry point: no saved/replayed model could request it, and cross-validation
could score only automatic smoothing. Phase 7 promotes the existing Gaussian
main-effects GAM through the same `StatisticalModelSpecification` and
`FittedStatisticalModel` seams. A new `.additiveGaussian` strategy owns an
`AdditiveModelSpecification` (selected terms, per-term span/degree, defaults,
robustness, convergence cap/tolerance). The strategy is intentionally explicit
and always Gaussian identity. In particular, integral-valued measurements do
not silently become a Poisson GAM; likelihood GAMs require a separately
validated IRLS/backfitting contract.

`CrossValidation` now refits that full specification inside every training
fold. Its scores remain Gaussian RMSE/MAE for this strategy, even for an
integral response, while automatic smoothing retains its binary/count routing.
This closes a particularly dangerous validation loophole: selecting a GAM's
terms or smoothing settings on rows that later appear in its held-out score.

Interpretability is exposed as data, not a view-only calculation.
`AdditiveTermDiagnostics` reports centered effect magnitude and approximate
per-term effective degrees of freedom; `AdditivePartialEffect` yields an
aligned predictor/effect/gradient curve. These are component-scale summaries
(not response predictions and not uncertainty intervals). The existing sigma
and total EDF remain approximate because independent LOESS traces are not a
joint GAM covariance calculation. Interaction terms, categorical effects,
likelihood GAMs, and simultaneous partial-effect uncertainty remain explicit
future work rather than silently implied by these plots.
