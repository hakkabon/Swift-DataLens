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
