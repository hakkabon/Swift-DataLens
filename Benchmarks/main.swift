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
