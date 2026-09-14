# Swift-DataLens

Local regression in pure Swift — `import DataLens`.

> Status: Cleveland-style LOESS (`Loess`, frozen), clean-room adaptive
> smoothing (`AdaptiveLoess`), and local likelihood (`LocalLikelihood`:
> Gaussian/Binomial/Poisson via Newton–IRLS). Least squares via LAPACK on
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
- **Batch evaluation + concurrency:** `predict(_:)` / `standardErrors(at:)`
  over query grids share one neighbor index (no per-call rebuilds);
  `*Concurrently` async variants and concurrent fits via indexed task
  groups — bit-identical to the sequential paths.

## Requirements

- Swift 6.1+ (swift-tools-version 6.1), strict concurrency enabled
- macOS 13+ / iOS 16+ / tvOS 16+ / watchOS 9+ / macCatalyst 16+ / Linux
- Xcode 16+ or SwiftPM CLI

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/hakkabon/Swift-DataLens.git", from: "0.4.0")
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

40 tests, all deterministic-or-seeded: `LoessTests` (ported 1:1, same
seeds/tolerances — exact linear/plane/quadratic reproduction, outlier
recovery, symmetry, SE/trace bounds, GCV span selection, invalid input),
`NearestNeighborTests` (kd-tree vs brute-force exact agreement on seeded
clouds with duplicates, coincident queries, degenerate inputs, mixed
tree/brute paths), `SolverSeamTests` (closed-form solves plus the
rank-deficient/singular/non-PD → nil contract, on whichever solver path
is active), `AdaptiveLoessTests` (exactness, outlier recovery, directly
observed adaptivity that beats the best fixed span, homogeneous parity),
`LocalLikelihoodTests` (Gaussian–Loess agreement, IRLS fixed point at
1e-9/1e-6, Binomial/Poisson recovery with deviance below null, separation
clamping, AIC span selection), `BatchTests` (batches equal pointwise
calls, concurrent variants bit-identical incl. fits) plus a version smoke
test (full suite ≈ 3s in debug).

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
│       ├── DataLens.swift
│       ├── LocalLikelihood.swift
│       ├── Loess.swift
│       ├── Internal
│       │   ├── Concurrency.swift (indexed task-group batch helper)
│       │   ├── Descriptive.swift (median)
│       │   ├── KDTree.swift (N-D kd-tree, exact brute-force parity)
│       │   ├── LinAlg.swift (Householder QR + least squares)
│       │   ├── LocalPolynomial.swift (shared WLS + leverage engine)
│       │   ├── NeighborSearch.swift (build-once routing: tree vs brute)
│       │   ├── Regression.swift (square solver)
│       │   └── SeededRNG.swift (test-only RNG + GaussianCache)
│       └── DataLens.docc
│           └── DataLens.md
└── Tests
    └── DataLensTests
        ├── AdaptiveLoessTests.swift
        ├── BatchTests.swift
        ├── DataLensTests.swift
        ├── LocalLikelihoodTests.swift
        ├── LoessTests.swift
        ├── NearestNeighborTests.swift
        └── SolverSeamTests.swift
```

## Known limitations / next steps

- Numerics are a vendored subset (QR + square solve + median + test RNG);
  the old repo keeps the full `LinAlg`/`Regression`/`Descriptive` suites.
  Divergences between the copies need a `docs/DECISIONS.md` entry.
- Next: automatic smoother choice (family + span in one call), derivative
  estimates, parallel span selection — smoothers' behavior stays frozen
  without a DECISIONS entry.
- Loader-style work is clean-room by policy (see `docs/DECISIONS.md`);
  contributions welcome (`swift build`, `swift test`,
  `swift run Benchmarks`, `swiftformat`, `swiftlint` before submitting).

## License

See `LICENSE`.
