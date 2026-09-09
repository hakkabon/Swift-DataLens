# Swift-Data-Lens

Local regression in pure Swift — `import DataLens`.

> Status: Cleveland-style LOESS ported (`Loess` byte-identical to
> `Numerical-Statistics`, over vendored QR least squares + square solver +
> median). Next: clean-room Loader-style adaptive smoothing and local
> likelihood. No GPL `locfit` code enters this repo.

## Features

- **LOESS (Cleveland 1979 / Cleveland–Devlin 1988):** local polynomials
  (deg 0–2, tricube weights, spans, 1-D + multivariate), bisquare
  robustness rounds, equivalent-kernel SEs, smoother trace, GCV span
  selection — self-contained, brute-force neighbors behind a kd-tree-ready
  seam.
- **Adaptive smoothing (Loader-style, clean-room):** variable bandwidths,
  local likelihood families — new types, no change to classic `Loess`
  without a `docs/DECISIONS.md` entry.

## Requirements

- Swift 6.1+ (swift-tools-version 6.1), strict concurrency enabled
- macOS 13+ / iOS 16+ / tvOS 16+ / watchOS 9+ / macCatalyst 16+ / Linux
- Xcode 16+ or SwiftPM CLI

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/hakkabon/Swift-Data-Lens.git", from: "0.1.0")
],
targets: [
    .target(name: "MyTarget", dependencies: ["DataLens"])
]
```

```bash
git clone https://github.com/hakkabon/Swift-Data-Lens.git
cd Swift-Data-Lens
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

10 tests, all deterministic-or-seeded: `LoessTests` (ported 1:1, same
seeds/tolerances — exact linear/plane/quadratic reproduction, outlier
recovery, symmetry, SE/trace bounds, GCV span selection, invalid input)
plus a version smoke test (full suite ≈ 1s in debug).

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
│       ├── DataLens.swift
│       ├── Loess.swift
│       ├── Internal
│       │   ├── Descriptive.swift (median)
│       │   ├── LinAlg.swift (Householder QR + least squares)
│       │   ├── Regression.swift (square solver)
│       │   └── SeededRNG.swift (test-only RNG + GaussianCache)
│       └── DataLens.docc
│           └── DataLens.md
└── Tests
    └── DataLensTests
        ├── DataLensTests.swift
        └── LoessTests.swift
```

## Known limitations / next steps

- Numerics are a vendored subset (QR + square solve + median + test RNG);
  the old repo keeps the full `LinAlg`/`Regression`/`Descriptive` suites.
  Divergences between the copies need a `docs/DECISIONS.md` entry.
- Next: clean-room Loader-style adaptive smoothing and local likelihood
  as new types (classic `Loess` behavior is frozen without a DECISIONS
  entry).
- Loader-style work is clean-room by policy (see `docs/DECISIONS.md`);
  contributions welcome (`swift build`, `swift test`,
  `swift run Benchmarks`, `swiftformat`, `swiftlint` before submitting).

## License

See `LICENSE`.
