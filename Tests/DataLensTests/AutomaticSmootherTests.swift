import Foundation
import Testing
@testable import DataLens

/// Tests for one-call automated tuning: response-type routing, GCV choice
/// between fixed and adaptive spans, graceful fallback, and validation.
@Suite("Automatic tuning")
struct AutomaticSmootherTests {
    func rmse(_ fitted: [Double], _ truth: [Double]) -> Double {
        sqrt(zip(fitted, truth).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(truth.count))
    }

    @Test func routesBinaryToBinomial() {
        var rng = SeedableRandomNumberGenerator(seed: 7001)
        let xs = (0..<60).map { _ in [Double.random(in: -2...2, using: &rng)] }
        let truth = xs.map { 1 / (1 + exp(-(1.5 * $0[0]))) }
        let ys = zip(xs, truth).map { Double.random(in: 0..<1, using: &rng) < $0.1 ? 1.0 : 0.0 }
        let (fit, summary) = AutomaticSmoother.fit(trainX: xs, trainY: ys)!
        guard case .likelihood(let ll) = fit else {
            Issue.record("expected likelihood fit, got \(fit)")
            return
        }
        #expect(ll.family == .binomial)
        #expect(rmse(fit.fittedValues, truth) < 0.2)
        #expect(summary.smoother == "LocalLikelihood")
        #expect(summary.detail.contains("Binomial"))
        #expect(summary.notes.isEmpty)
        #expect(summary.description.contains("Binomial"))
    }

    @Test func routesCountsToPoisson() {
        var rng = SeedableRandomNumberGenerator(seed: 7002)
        let xs = (0..<60).map { [Double($0) / 20] }
        let truth = xs.map { exp(0.3 + 0.5 * $0[0]) }
        var ys: [Double] = []
        for t in truth {
            let l = exp(-t)
            var k = 0
            var p = 1.0
            repeat {
                k += 1
                p *= Double.random(in: 0..<1, using: &rng)
            } while p > l
            ys.append(Double(k - 1))
        }
        let (fit, summary) = AutomaticSmoother.fit(trainX: xs, trainY: ys)!
        guard case .likelihood(let ll) = fit else {
            Issue.record("expected likelihood fit, got \(fit)")
            return
        }
        #expect(ll.family == .poisson)
        #expect(rmse(fit.fittedValues, truth) < 1.0)
        #expect(summary.detail.contains("Poisson"))
    }

    @Test func heterogeneousPrefersAdaptive() {
        // Same bump truth as AdaptiveLoessTests: adaptivity wins on GCV.
        let n = 60
        let xs = (0..<n).map { [Double($0) / 10] }
        func truth(_ x: Double) -> Double {
            let t = max(0, x - 3)
            return t * t * sin(2 * t) * 0.5
        }
        let truthVals = xs.map { truth($0[0]) }
        var rng = SeedableRandomNumberGenerator(seed: 7003)
        var cache = GaussianCache()
        let ys = xs.map { truth($0[0]) + 0.05 * cache.nextStandardNormal(using: &rng) }
        let (fit, summary) = AutomaticSmoother.fit(trainX: xs, trainY: ys, degree: 2)!
        guard case .adaptive = fit else {
            Issue.record("expected adaptive fit, got \(fit)")
            return
        }
        #expect(summary.smoother == "AdaptiveLoess")
        #expect(rmse(fit.fittedValues, truthVals) < 0.15)
    }

    @Test func homogeneousPicksEitherContinuous() {
        var rng = SeedableRandomNumberGenerator(seed: 7004)
        var cache = GaussianCache()
        let xs = (0..<50).map { [Double($0) / 10] }
        let truthVals = xs.map { sin($0[0]) }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let (fit, summary) = AutomaticSmoother.fit(trainX: xs, trainY: ys, degree: 1)!
        switch fit {
        case .loess, .adaptive:
            break
        case .likelihood:
            Issue.record("continuous data must not route to likelihood")
        case .nadarayaWatson:
            Issue.record("tuner never routes kernel fits; selection stays explicit")
        case .whittakerEilers:
            Issue.record("tuner never routes penalized fits; selection stays explicit")
        }
        #expect(rmse(fit.fittedValues, truthVals) < 0.2)
        #expect(summary.description.contains("GCV"))
        // Batch + single-point forwarding agree.
        let grid = [[1.0], [2.0], [3.0]]
        #expect(fit.predict(grid) == grid.map { fit.predict($0) })
    }

    @Test func tinyInputFallsBackWithNote() {
        // n=3, degree 2, non-integral responses: AdaptiveLoess has no valid
        // neighborhood (needs k ≥ 5), so the tuner falls back to fixed-span
        // Loess and says so. (Integral responses would route to Poisson.)
        let xs = [[0.0], [1.0], [2.0]]
        let ys = [0.0, 1.5, 4.0]
        let (fit, summary) = AutomaticSmoother.fit(trainX: xs, trainY: ys, degree: 2)!
        guard case .loess = fit else {
            Issue.record("expected loess fallback, got \(fit)")
            return
        }
        #expect(!summary.notes.isEmpty)
        #expect(summary.notes.joined().contains("AdaptiveLoess"))
    }

    @Test func invalidInput() {
        #expect(AutomaticSmoother.fit(trainX: [], trainY: []) == nil)
        #expect(AutomaticSmoother.fit(trainX: [[0]], trainY: [1, 2]) == nil)
        #expect(AutomaticSmoother.fit(trainX: [[0]], trainY: [1], degree: 5) == nil)
        #expect(AutomaticSmoother.fit(trainX: [[0]], trainY: [0.5], degree: 1) == nil)
    }

    @Test func shallowTuningSkipsAdaptiveContender() {
        // Continuous sine truth: shallow selects fixed-span Loess, says
        // the adaptive leg was skipped, and still recovers the curve.
        var rng = SeedableRandomNumberGenerator(seed: 7010)
        var cache = GaussianCache()
        let xs = (0..<60).map { [Double($0) / 10] }
        let truth = xs.map { sin($0[0]) }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let (fit, summary) = AutomaticSmoother.fit(trainX: xs, trainY: ys, adaptiveContender: false)!
        guard case .loess = fit else {
            Issue.record("expected loess fit, got \(fit)")
            return
        }
        #expect(summary.smoother == "Loess")
        #expect(summary.reason.contains("adaptive contender disabled"))
        #expect(summary.notes.contains(
            "Adaptive contender disabled (shallow tuning); comparing fixed spans only."))
        #expect(rmse(fit.fittedValues, truth) < 0.15)
    }

    @Test func shallowTuningPreservesRouting() {
        // The flag only trims the continuous competition: binary data
        // still routes to binomial local likelihood.
        var rng = SeedableRandomNumberGenerator(seed: 7011)
        let xs = (0..<60).map { _ in [Double.random(in: -2...2, using: &rng)] }
        let truth = xs.map { 1 / (1 + exp(-(1.5 * $0[0]))) }
        let ys = zip(xs, truth).map { Double.random(in: 0..<1, using: &rng) < $0.1 ? 1.0 : 0.0 }
        let (fit, summary) = AutomaticSmoother.fit(trainX: xs, trainY: ys, adaptiveContender: false)!
        guard case .likelihood(let ll) = fit else {
            Issue.record("expected likelihood fit, got \(fit)")
            return
        }
        #expect(ll.family == .binomial)
        #expect(summary.smoother == "LocalLikelihood")
    }
}
