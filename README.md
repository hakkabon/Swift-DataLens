# Swift-DataLens

Local regression in pure Swift — `import DataLens`.

> Status: Cleveland-style LOESS (`Loess`, frozen), Nadaraya–Watson kernel
> regression (`NadarayaWatson`), Whittaker–Eilers penalized smoothing
> (`WhittakerEilers`, incl. Hodrick–Prescott as order 2), total-variation
> denoising (`TotalVariation`, 1-D fused lasso), clean-room adaptive
> smoothing (`AdaptiveLoess`), local likelihood (`LocalLikelihood`),
> batch evaluation, one-call tuning, plus derivatives, explicit
> extrapolation, and missing-data handling. Least squares via LAPACK on
> Apple with a vendored fallback; neighbors via kd-tree or brute force.
> No GPL `locfit` code enters this repo.

## Features

- **LOESS (Cleveland 1979 / Cleveland–Devlin 1988):** local polynomials
  (deg 0–2, tricube weights, spans, 1-D + multivariate), bisquare
  robustness rounds, equivalent-kernel SEs, smoother trace, GCV span
  selection — self-contained, neighbors via a build-once kd-tree (large n,
  small spans) or brute force, exactly equal on every path; least squares
  via LAPACK (`NumericCoreAccelerate`) on Apple platforms with a vendored
  Householder fallback elsewhere.
- **Nadaraya–Watson kernel regression (`NadarayaWatson`):** locally
  constant tricube fits with the same bisquare robustness, SEs, trace,
  and GCV span selection as `Loess` — bit-identical to
  `Loess.fit(degree: 0)` by construction (pinned), plus analytic
  tricube gradients. Release bench: 2.8 ms fit / 1.4 ms x200 grid
  (n=100; no QR solves anywhere).
- **Whittaker–Eilers penalized smoothing (`WhittakerEilers`, Eilers
  2003):** ŷ = (I + λDᵀD)⁻¹y over the x-ordered sequence (order-2 ==
  Hodrick–Prescott), solved banded in O(n·d²) with exact trace and
  standard errors via Takahashi — plus GCV λ selection, interpolation
  prediction, and segment-slope gradients. Release bench: 0.5 ms fit /
  microseconds x200 grid (n=100). Sequence semantics: x orders,
  spacing is ignored — bin first for wild grids.
- **Total-variation denoising (`TotalVariation`, 1-D fused lasso):**
  piecewise-constant fits minimizing ½‖x−y‖² + λ‖Dx‖₁ via ADMM
  (banded direct solve per round), KKT-verified in tests, GCV λ
  selection over the segment-count trace, nearest-neighbor prediction,
  homoskedastic σ bands. The edge-preserving complement to the smooth
  family. Release bench: 2.8 ms fit / microseconds grid (n=100).
- **Adaptive smoothing (`AdaptiveLoess`, clean-room Loader-style):**
  per-point AICc neighborhood selection (flat stretches average ~2× the
  neighborhoods of curvy ones; beats the best fixed span ~6× on
  heterogeneous truth), same bisquare robustness, SEs, trace and GCV-style
  diagnostics as `Loess` — no change to classic `Loess` without a
  `docs/DECISIONS.md` entry.
- **Local likelihood (`LocalLikelihood`, clean-room Loader-style):**
  Gaussian (identity), Binomial (logit), Poisson (log) families by
  Newton–IRLS with step-halving (≤25 rounds, 1e-8 tolerance), SPD systems
  through the Cholesky seam; boundary MLEs saturate finitely, deviances +
  AIC span selection, delta-method SEs.
- **Typed diagnostics:** every `FittedSmoother` exposes serializable
  `FitDiagnostics` (family, link, effective degrees of freedom, scale,
  deviance) and raw, Pearson, and family-correct deviance residuals.
- **Solver conformance contract:** the built-in fallback and
  Swift-NumericCore run an identical checked-in fixture suite, including
  numerical answers and singular-matrix verdicts.
- **Batch evaluation + concurrency:** `predict(_:)` / `standardErrors(at:)`
  over query grids share one neighbor index (no per-call rebuilds);
  `*Concurrently` async variants and concurrent fits via indexed task
  groups — bit-identical to the sequential paths, cooperatively
  cancellable.
- **Automated tuning (`AutomaticSmoother`):** routes by response type
  (binary → binomial, counts → Poisson, else continuous), tunes spans by
  AIC/GCV with adaptive-vs-fixed competition, and reports what it chose
  and why in a printable `TuningSummary` — fallbacks noted, never silent.
  `adaptiveContender: false` skips the adaptive leg for shallow
  interactive tuning (same routing, fixed-span Loess only).
- **Predictive flexibility:** gradients (slopes/trends) on all smoothers,
  explicit extrapolation policies (polynomial/nearest/unavailable) on every
  predict/SE path, and missing-data dropping with survivor indices.
- **Fast adaptive grids (`predictFast` / `standardErrorsFast`):** the
  adaptive smoother reuses each query's nearest training point's
  AICc-selected neighborhood instead of re-selecting — ~25× faster
  means, ~45× faster SEs on the release bench (n=60, x200 grid),
  agreeing with the exact paths within half the noise scale.

## Requirements

- Swift 6.1+ (swift-tools-version 6.1), strict concurrency enabled
- macOS 13+ / iOS 16+ / tvOS 16+ / watchOS 9+ / macCatalyst 16+ / Linux
- Xcode 16+ or SwiftPM CLI
- Swift-NumericCore 0.3.x (which consumes the tagged Rust-NumericCore 0.3.0
  XCFramework by checksum)

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/hakkabon/Swift-DataLens.git", from: "0.6.0")
],
targets: [
    .target(name: "MyTarget", dependencies: ["DataLens"])
]
```

```bash
git clone https://github.com/hakkabon/Swift-DataLens.git
cd Swift-DataLens
swift build
swift test
```

```swift
import DataLens

let xs = (0..<15).map { [Double($0)] }
let ys = xs.map { 2 * $0[0] - 1 }
let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 1)!
fit.predict([7.5]) // 14.0
fit.sigma          // residual scale
fit.trace          // smoother-matrix trace (effective df)
Loess.selectSpan(trainX: xs, trainY: ys, spans: [0.4, 0.6, 0.8], degree: 1)
```

## Testing

```bash
swift test
swift test --filter DataLensTests
```

99 tests, all deterministic-or-seeded: `LoessTests` (ported 1:1, same
seeds/tolerances — exact linear/plane/quadratic reproduction, outlier
recovery, symmetry, SE/trace bounds, GCV span selection, span
selection on dropped rows, invalid input),
`NearestNeighborTests` (kd-tree vs brute-force exact agreement on seeded
clouds with duplicates, coincident queries, degenerate inputs, mixed
tree/brute paths), `SolverSeamTests` (closed-form solves plus the
rank-deficient/singular/non-PD → nil contract on both solver paths, with
the Linux fallback Cholesky pinned directly), `AdaptiveLoessTests` (exactness, outlier recovery, directly
observed adaptivity that beats the best fixed span, homogeneous parity,
fast-path agreement within half the noise scale),
`NadarayaWatsonTests` (constant reproduction, bit-parity with degree-0
Loess, sine recovery, batch/concurrent agreement, analytic-vs-numeric
gradients, missing-data masks, invalid input),
`WhittakerEilersTests` (λ = 0 interpolation, exact lines, GCV recovery,
batch/concurrent parity, order restoration on shuffled input,
segment-slope gradients, missing masks, carrier round-trip, invalid
input), `TotalVariationTests` (λ = 0 reproduction, step recovery,
KKT optimality on steps and sine, batch/concurrent parity, missing
masks, carrier round-trip, invalid input),
`BandedSPDTests` (factor/solve/inverse vs dense references,
verdict parity on non-SPD),`LocalLikelihoodTests` (Gaussian–Loess agreement, IRLS fixed point at
1e-9/1e-6, Binomial/Poisson recovery with deviance below null, separation
clamping, AIC span selection), `BatchTests` (batches equal pointwise
calls, concurrent variants bit-identical incl. fits, cooperative
cancellation with abort/completion tests),
`AutomaticSmootherTests` (response-type routing, adaptive win on GCV,
shallow flag skips the contender with routing intact,
fallback with a recorded note, invalid input), `FlexibilityTests`
(derivative exactness + cosine tracking, extrapolation policies on values
and SEs, missing-data masks with bit-identical clean fits) plus a version
smoke test (full suite ≈ 4s in debug).

```bash
swift run Benchmarks # micro-benchmarks (debug numbers; compare relatively)
```

## Project structure

```
.
├── AGENTS.md
├── Benchmarks
│   └── main.swift
├── LICENSE
├── Package.swift
├── README.md
├── docs
│   └── DECISIONS.md
├── Sources
│   └── DataLens
│       ├── AdaptiveLoess.swift
│       ├── AutomaticSmoother.swift
│       ├── DataLens.swift
│       ├── ExtrapolationPolicy.swift
│       ├── LocalLikelihood.swift
│       ├── Loess.swift
│       ├── Internal
│       │   ├── Concurrency.swift (indexed task-group batch helper)
│       │   ├── Descriptive.swift (median)
│       │   ├── KDTree.swift (N-D kd-tree, exact brute-force parity)
│       │   ├── LinAlg.swift (Householder QR + least squares)
│       │   ├── LocalPolynomial.swift (shared WLS + leverage engine)
│       │   ├── MissingData.swift (non-finite row dropping + masks)
│       │   ├── NeighborSearch.swift (build-once routing: tree vs brute)
│       │   ├── Regression.swift (square solver)
│       │   └── SeededRNG.swift (test-only RNG + GaussianCache)
│       └── DataLens.docc
│           └── DataLens.md
└── Tests
    └── DataLensTests
        ├── AdaptiveLoessTests.swift
        ├── AutomaticSmootherTests.swift
        ├── BatchTests.swift
        ├── DataLensTests.swift
        ├── FlexibilityTests.swift
        ├── LocalLikelihoodTests.swift
        ├── LoessTests.swift
        ├── NearestNeighborTests.swift
        └── SolverSeamTests.swift
```

## Known limitations / next steps

- Numerics are a vendored subset (QR + square solve + median + test RNG);
  the old repo keeps the full `LinAlg`/`Regression`/`Descriptive` suites.
  Divergences between the copies need a `docs/DECISIONS.md` entry.
- Next: adaptive-bandwidth likelihood, robustness reweighting for
  likelihood families, further families — smoothers' behavior stays frozen
  without a DECISIONS entry.
- Loader-style work is clean-room by policy (see `docs/DECISIONS.md`);
  contributions welcome (`swift build`, `swift test`,
  `swift run Benchmarks`, `swiftformat`, `swiftlint` before submitting).

## License

See `LICENSE`.
