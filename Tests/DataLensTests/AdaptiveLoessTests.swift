import Foundation
import Testing
@testable import DataLens

/// Tests for clean-room adaptive smoothing: exactness where the truth is
/// simple, and directly observed adaptivity (larger neighborhoods on flat
/// stretches, smaller ones where the truth curves) where it is not.
@Suite("Adaptive LOESS")
struct AdaptiveLoessTests {
    func rmse(_ fitted: [Double], _ truth: [Double]) -> Double {
        sqrt(zip(fitted, truth).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(truth.count))
    }

    @Test func reproducesLinear() {
        // Degree 1 reproduces lines exactly under every neighborhood.
        let xs = (0..<15).map { [Double($0)] }
        let ys = xs.map { 2 * $0[0] - 1 }
        let fit = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1)!
        for (f, y) in zip(fit.fittedValues, ys) {
            #expect(abs(f - y) <= 1e-9)
        }
        #expect(abs(fit.predict([7.5]) - 14.0) <= 1e-9)
    }

    @Test func adaptsToHeterogeneousCurvature() {
        // Flat left half, growing oscillations on the right (smooth onset).
        let n = 60
        let xs = (0..<n).map { [Double($0) / 10] }
        func truth(_ x: Double) -> Double {
            let t = max(0, x - 3)
            return t * t * sin(2 * t) * 0.5
        }
        let truthVals = xs.map { truth($0[0]) }
        var rng = SeedableRandomNumberGenerator(seed: 4242)
        var cache = GaussianCache()
        let ys = xs.map { truth($0[0]) + 0.05 * cache.nextStandardNormal(using: &rng) }
        let fit = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 2)!
        // Adaptivity, observed directly: flat stretches average larger
        // neighborhoods than curvy ones.
        var flatSum = 0, flatCount = 0, wigglySum = 0, wigglyCount = 0
        for (i, k) in fit.selectedNeighborhoods.enumerated() {
            if xs[i][0] < 2.5 { flatSum += k; flatCount += 1 }
            if xs[i][0] > 3.5 { wigglySum += k; wigglyCount += 1 }
        }
        let flatMean = Double(flatSum) / Double(flatCount)
        let wigglyMean = Double(wigglySum) / Double(wigglyCount)
        #expect(flatMean >= 1.5 * wigglyMean)
        // ...and the adaptive fit is competitive with the best fixed span.
        var bestFixed = Double.infinity
        for span in [0.3, 0.5, 0.7, 0.9] {
            let fixed = Loess.fit(trainX: xs, trainY: ys, span: span, degree: 2)!
            bestFixed = min(bestFixed, rmse(fixed.fittedValues, truthVals))
        }
        #expect(rmse(fit.fittedValues, truthVals) <= bestFixed)
    }

    @Test func nearParityOnHomogeneous() {
        // On uniformly wiggly truth, adaptivity costs little.
        var rng = SeedableRandomNumberGenerator(seed: 4243)
        var cache = GaussianCache()
        let xs = (0..<50).map { [Double($0) / 10] }
        let truthVals = xs.map { sin($0[0]) }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let adaptive = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1)!
        let fixed = Loess.fit(trainX: xs, trainY: ys, span: 0.4, degree: 1)!
        #expect(rmse(adaptive.fittedValues, truthVals) <= 1.2 * rmse(fixed.fittedValues, truthVals))
    }

    @Test func robustToOutlier() {
        var ys = (0..<20).map { 2 * Double($0) }
        ys[10] = 500
        let xs = (0..<20).map { [Double($0)] }
        let fit = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1)!
        #expect(abs(fit.fittedValues[10] - 20.0) <= 3.0)
    }

    @Test func standardErrors() {
        var rng = SeedableRandomNumberGenerator(seed: 4244)
        var cache = GaussianCache()
        let xs = (0..<50).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fit = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1)!
        #expect(fit.sigma > 0.05)
        #expect(fit.sigma < 0.3)
        for x in [[1.0], [3.0]] {
            let se = fit.standardError(at: x)!
            #expect(se > 0)
            #expect(se < 0.5)
        }
    }

    @Test func invalidInput() {
        #expect(AdaptiveLoess.fit(trainX: [], trainY: []) == nil)
        #expect(AdaptiveLoess.fit(trainX: [[0]], trainY: [1, 2]) == nil)
        #expect(AdaptiveLoess.fit(trainX: [[0]], trainY: [1], degree: 3) == nil)
        // Too few points for any valid neighborhood (degree 2 needs k ≥ 5).
        let xs = (0..<4).map { [Double($0)] }
        #expect(AdaptiveLoess.fit(trainX: xs, trainY: [0, 1, 2, 3], degree: 2) == nil)
        // All candidates below the minimum size (degree 1 needs k ≥ 4).
        let xs15 = (0..<15).map { [Double($0)] }
        let ys15 = xs15.map { $0[0] }
        #expect(AdaptiveLoess.fit(trainX: xs15, trainY: ys15, degree: 1,
                                  neighborhoods: [2, 3]) == nil)
    }

    @Test func fastPredictionAgreesWithExact() {
        // Sine truth + seeded noise; explicit candidate grid keeps the
        // debug suite fast while still exercising borrowed bandwidths.
        let n = 80
        let xs = (0..<n).map { [Double($0) / 10] }
        var rng = SeedableRandomNumberGenerator(seed: 4245)
        var cache = GaussianCache()
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fit = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 2,
                                    neighborhoods: [12, 25, 50, 80],
                                    robustIterations: 1)!
        let grid = (0..<100).map { [Double($0) / 100 * 7.9] }
        let exact = fit.predict(grid)
        let fast = fit.predictFast(grid)
        var maxDiff = 0.0
        for (a, b) in zip(exact, fast) {
            #expect(a.isFinite && b.isFinite)
            maxDiff = max(maxDiff, abs(a - b))
        }
        // Borrowed bandwidths stay within half the noise scale
        // (pinned: 0.039 observed at seed 4245).
        #expect(maxDiff <= 0.05)
        // Batch fast equals single fast on identical bits.
        #expect(fit.predictFast([grid[50]]) == [fast[50]])
        // SEs agree where both are available.
        let exactSE = fit.standardErrors(at: grid)
        let fastSE = fit.standardErrorsFast(at: grid)
        var maxSEDiff = 0.0
        for (a, b) in zip(exactSE, fastSE) {
            guard let a, let b else {
                #expect(a == nil && b == nil)
                continue
            }
            maxSEDiff = max(maxSEDiff, abs(a - b))
        }
        // SE agreement at the same scale (pinned: 0.037 observed).
        #expect(maxSEDiff <= 0.05)
    }

    @Test func fastConcurrentMatchesFastBatch() async throws {
        let n = 40
        let xs = (0..<n).map { [Double($0) / 10] }
        var rng = SeedableRandomNumberGenerator(seed: 4246)
        var cache = GaussianCache()
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fit = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1,
                                    neighborhoods: [8, 20, 40],
                                    robustIterations: 0)!
        let grid = (0..<50).map { [Double($0) / 50 * 3.9] }
        #expect(try await fit.predictFastConcurrently(grid) == fit.predictFast(grid))
        #expect(try await fit.standardErrorsFastConcurrently(at: grid)
            == fit.standardErrorsFast(at: grid))
    }

    @Test func fastMatchesExactOnDuplicatedInputs() {
        // Rank-deficient neighborhoods degrade through the same bounded
        // fallback cascade on both paths, so they agree to solver noise.
        let xs = [[0.0], [1.0], [1.0], [1.0], [2.0], [3.0]]
        let ys = [0.0, 1.0, 1.5, 0.5, 2.0, 3.0]
        let fit = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1)!
        let grid = [[0.5], [1.0], [2.5]]
        for (a, b) in zip(fit.predictFast(grid), fit.predict(grid)) {
            #expect(abs(a - b) <= 1e-9)
        }
    }
}
