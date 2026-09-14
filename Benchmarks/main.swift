import DataLens
import Foundation

/// Micro-benchmarks: `swift run -c release Benchmarks`.
/// Uses `Date` for timing so it works on macOS and Linux.
/// Ported from Numerical-Statistics' LOESS section (deterministic sine data,
/// so no seeded RNG needed here).
func bench(_ label: String, iterations: Int = 100_000, warmup: Int = 1000, _ body: () -> Void) {
    // Warm up (kept small for heavy fits like loess).
    for _ in 0..<warmup { body() }
    let start = Date()
    for _ in 0..<iterations { body() }
    let ms = Date().timeIntervalSince(start) * 1000
    let padded = label.padding(toLength: 28, withPad: " ", startingAt: 0)
    let nsPerDraw = ms * 1e6 / Double(iterations)
    print(String(format: "%@ %8.1f ms / %d draws (%6.0f ns/draw)", padded, ms, iterations, nsPerDraw))
}

print("DataLens \(DataLens.version)")
// Loess fit (n=100, degree 2, 2 robust rounds) and batched prediction.
let loessX = (0..<100).map { [Double($0) / 20] }
let loessY = loessX.map { sin($0[0]) }
let loessFit = Loess.fit(trainX: loessX, trainY: loessY, span: 0.4, degree: 2,
                         robustIterations: 2)!
bench("loess fit n=100", iterations: 5, warmup: 1) {
    _ = Loess.fit(trainX: loessX, trainY: loessY, span: 0.4, degree: 2,
                  robustIterations: 2)
}
bench("loess predict x50", iterations: 20, warmup: 2) {
    var s = 0.0
    for i in 0..<50 { s += loessFit.predict([Double(i) / 10]) }
    _ = s
}
// Adaptive fit (n=60, degree 2, default grid + 2 robust rounds).
let adaptX = (0..<60).map { [Double($0) / 10] }
let adaptY = adaptX.map { sin($0[0]) }
bench("adaptive fit n=60", iterations: 2, warmup: 0) {
    _ = AdaptiveLoess.fit(trainX: adaptX, trainY: adaptY, degree: 2,
                          robustIterations: 2)
}
// Local-binomial fit (n=80, degree 1, span 0.5; deterministic labels).
let llX = (0..<80).map { [Double($0) / 20 - 2] }
let llY = llX.map { sin($0[0] * 2) > 0 ? 1.0 : 0.0 }
bench("local-binomial fit n=80", iterations: 2, warmup: 0) {
    _ = LocalLikelihood.fit(trainX: llX, trainY: llY, degree: 1,
                            family: .binomial, span: 0.5)
}
// Batch prediction over a 200-point grid: shared index vs per-call loop.
let gridX = (0..<200).map { [Double($0) / 40] }
bench("loess batch predict x200", iterations: 5, warmup: 1) {
    _ = loessFit.predict(gridX)
}
// One-call automated tuning (n=60 continuous: adaptive + 3 fixed spans).
bench("auto tune n=60", iterations: 1, warmup: 0) {
    _ = AutomaticSmoother.fit(trainX: adaptX, trainY: adaptY, degree: 2)
}
