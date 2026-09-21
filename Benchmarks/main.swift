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
// Adaptive grid prediction (n=60 fit, x200 grid): fresh AICc selection
// per query vs borrowed training-point bandwidths.
let adaptFit60 = AdaptiveLoess.fit(trainX: adaptX, trainY: adaptY, degree: 2,
                                   robustIterations: 2)!
let adaptGrid = (0..<200).map { [Double($0) / 200 * 6] }
bench("adaptive exact x200", iterations: 3, warmup: 0) {
    _ = adaptFit60.predict(adaptGrid)
}
bench("adaptive fast x200", iterations: 3, warmup: 0) {
    _ = adaptFit60.predictFast(adaptGrid)
}
bench("adaptive SE exact x200", iterations: 3, warmup: 0) {
    _ = adaptFit60.standardErrors(at: adaptGrid)
}
bench("adaptive SE fast x200", iterations: 3, warmup: 0) {
    _ = adaptFit60.standardErrorsFast(at: adaptGrid)
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
// Nadaraya-Watson fit (n=100, span 0.4, 2 robust rounds) and grid.
let nwX = (0..<100).map { [Double($0) / 20] }
let nwY = nwX.map { sin($0[0]) }
let nwFit = NadarayaWatson.fit(trainX: nwX, trainY: nwY, span: 0.4, robustIterations: 2)!
bench("nw fit n=100", iterations: 5, warmup: 1) {
    _ = NadarayaWatson.fit(trainX: nwX, trainY: nwY, span: 0.4, robustIterations: 2)
}
bench("nw predict x200", iterations: 5, warmup: 1) {
    _ = nwFit.predict(gridX)
}
// Whittaker fit (n=100, λ=100, order 2) and grid.
let weX = (0..<100).map { [Double($0) / 20] }
let weY = weX.map { sin($0[0]) }
let weFit = WhittakerEilers.fit(trainX: weX, trainY: weY, lambda: 100)!
bench("we fit n=100", iterations: 5, warmup: 1) {
    _ = WhittakerEilers.fit(trainX: weX, trainY: weY, lambda: 100)
}
bench("we predict x200", iterations: 5, warmup: 1) {
    _ = weFit.predict(gridX)
}
// Total-variation fit (n=100, λ=0.5) and grid.
let tvX = (0..<100).map { [Double($0) / 20] }
let tvY = tvX.map { sin($0[0]) }
let tvFit = TotalVariation.fit(trainX: tvX, trainY: tvY, lambda: 0.5)!
bench("tv fit n=100", iterations: 5, warmup: 1) {
    _ = TotalVariation.fit(trainX: tvX, trainY: tvY, lambda: 0.5)
}
bench("tv predict x200", iterations: 5, warmup: 1) {
    _ = tvFit.predict(gridX)
}
// One-call automated tuning (n=60 continuous: adaptive + 3 fixed spans).
bench("auto tune n=60", iterations: 1, warmup: 0) {
    _ = AutomaticSmoother.fit(trainX: adaptX, trainY: adaptY, degree: 2)
}
// Same call with the adaptive contender skipped (shallow tuning).
bench("auto tune shallow n=60", iterations: 1, warmup: 0) {
    _ = AutomaticSmoother.fit(trainX: adaptX, trainY: adaptY, degree: 2,
                              adaptiveContender: false)
}
// Gradient grid (x200): slopes alongside the smoother.
bench("loess gradient x200", iterations: 5, warmup: 1) {
    _ = loessFit.gradients(at: gridX)
}

// Large sparse statistical workload: four treatment-coded factors create
// 253 coefficients but at most four non-intercept entries per row. This is the
// representative geometry for workbench filters/segment comparisons, where a
// sparse CGLS bridge can beat materializing a dense augmented QR design.
let factorLevels = 64
let factorRows = 12_000
let factorX = (0..<factorRows).map { row in
    (0..<4).map { factor in Double((row / (factor + 1) + factor * 17) % factorLevels) }
}
let factorY = factorX.map { row in
    1 + 0.2 * row[0] / 63 - 0.15 * row[1] / 63 + 0.1 * row[2] / 63 - 0.05 * row[3] / 63
}
let factorSpecification = MultivariateModelSpecification(
    terms: (0..<4).map {
        .categorical(CategoricalTermSpecification(
            predictorIndex: $0, levels: Array(0..<factorLevels), referenceLevel: 0
        ))
    },
    penaltyWeight: 0.1, maxIterations: 40, tolerance: 1e-8
)
bench("multivariate factors n=12000", iterations: 1, warmup: 0) {
    _ = MultivariateModel.fit(
        trainX: factorX, trainY: factorY, family: .gaussian,
        specification: factorSpecification
    )
}
