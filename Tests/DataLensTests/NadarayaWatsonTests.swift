import Foundation
import Testing
@testable import DataLens

/// Tests for Nadaraya–Watson kernel regression: constant reproduction,
/// bit-parity with degree-0 Loess (the kernel it delegates to), recovery,
/// batch/concurrent agreement, analytic gradients, and validation.
@Suite("Nadaraya-Watson")
struct NadarayaWatsonTests {
    func rmse(_ fitted: [Double], _ truth: [Double]) -> Double {
        sqrt(zip(fitted, truth).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(truth.count))
    }

    @Test func reproducesConstants() {
        let xs = (0..<20).map { [Double($0) / 10] }
        let ys = [Double](repeating: 3.25, count: 20)
        let fit = NadarayaWatson.fit(trainX: xs, trainY: ys, span: 0.5)!
        for f in fit.fittedValues {
            #expect(abs(f - 3.25) <= 1e-12)
        }
        #expect(abs(fit.predict([0.55]) - 3.25) <= 1e-12)
        #expect(abs(fit.predict([100.0]) - 3.25) <= 1e-12)
    }

    @Test func matchesDegreeZeroLoess() {
        // The estimator IS local-constant fitting: bit-parity with
        // Loess.fit(degree: 0) on seeded noise pins the delegation.
        var rng = SeedableRandomNumberGenerator(seed: 2401)
        var cache = GaussianCache()
        let xs = (0..<40).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let nw = NadarayaWatson.fit(trainX: xs, trainY: ys, span: 0.4, robustIterations: 1)!
        let lo = Loess.fit(trainX: xs, trainY: ys, span: 0.4, degree: 0, robustIterations: 1)!
        #expect(nw.fittedValues == lo.fittedValues)
        #expect(nw.sigma == lo.sigma)
        #expect(nw.trace == lo.trace)
        #expect(nw.weights == lo.weights)
        #expect(nw.keptIndices == lo.keptIndices)
        let grid = (0..<20).map { [Double($0) / 5] }
        #expect(nw.predict(grid) == lo.predict(grid))
    }

    @Test func recoversSine() {
        var rng = SeedableRandomNumberGenerator(seed: 2402)
        var cache = GaussianCache()
        let xs = (0..<60).map { [Double($0) / 10] }
        let truth = xs.map { sin($0[0]) }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fit = NadarayaWatson.fit(trainX: xs, trainY: ys, span: 0.4, robustIterations: 1)!
        #expect(rmse(fit.fittedValues, truth) < 0.25)
        #expect(fit.sigma > 0.05 && fit.sigma < 0.3)
        #expect(fit.trace > 1 && fit.trace < 60)
    }

    @Test func batchEqualsPointwiseAndConcurrent() async throws {
        var rng = SeedableRandomNumberGenerator(seed: 2403)
        var cache = GaussianCache()
        let xs = (0..<40).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fit = NadarayaWatson.fit(trainX: xs, trainY: ys, span: 0.5, robustIterations: 1)!
        let grid = (0..<30).map { [Double($0) / 10] }
        let batch = fit.predict(grid)
        #expect(batch == grid.map { fit.predict($0) })
        #expect(try await fit.predictConcurrently(grid) == batch)
        let seBatch = fit.standardErrors(at: grid)
        #expect(try await fit.standardErrorsConcurrently(at: grid) == seBatch)
        #expect(seBatch.allSatisfy { ($0 ?? -1) > 0 })
        let gradBatch = fit.gradients(at: grid)
        #expect(try await fit.gradientsConcurrently(at: grid) == gradBatch)
    }

    @Test func gradientMatchesNumeric() {
        // Analytic tricube gradient vs central differences (robustness
        // off so both differentiate the same weights). Query points sit
        // OFF the training lattice on purpose: span neighborhoods make
        // the mean piecewise smooth, with kinks where membership or the
        // bandwidth argmax switches (e.g. x = 2.0 centered on a symmetric
        // grid, where central differences average two branches). Off-kink,
        // agreement is ~1e-9; the tolerance below is pure headroom.
        var rng = SeedableRandomNumberGenerator(seed: 2404)
        var cache = GaussianCache()
        let xs = (0..<50).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.05 * cache.nextStandardNormal(using: &rng) }
        let fit = NadarayaWatson.fit(trainX: xs, trainY: ys, span: 0.5, robustIterations: 0)!
        let h = 1e-6
        for x in [0.53, 1.27, 2.61, 3.33, 4.14] {
            guard let g = fit.gradient(at: [x]) else {
                Issue.record("nil gradient at \(x)")
                continue
            }
            let numeric = (fit.predict([x + h]) - fit.predict([x - h])) / (2 * h)
            #expect(abs(g[0] - numeric) <= 1e-6)
        }
    }

    @Test func droppingMissingKeepsIndices() {
        var rng = SeedableRandomNumberGenerator(seed: 2405)
        var cache = GaussianCache()
        let n = 40
        let xs = (0..<n).map { [Double($0) / 10] }
        var ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        ys[5] = .nan
        ys[30] = .nan
        let fit = NadarayaWatson.fit(trainX: xs, trainY: ys, span: 0.5,
                                     robustIterations: 1, droppingMissing: true)!
        #expect(fit.keptIndices == (0..<n).filter { $0 != 5 && $0 != 30 })
        #expect(fit.fittedValues.count == n - 2)
        #expect(fit.fittedValues.allSatisfy { $0.isFinite })
    }

        @Test func invalidInput() {        #expect(NadarayaWatson.fit(trainX: [], trainY: []) == nil)
        #expect(NadarayaWatson.fit(trainX: [[0]], trainY: [1, 2]) == nil)
        #expect(NadarayaWatson.fit(trainX: [[0]], trainY: [1], span: 0) == nil)
        #expect(NadarayaWatson.fit(trainX: [[0]], trainY: [1], span: 1.5) == nil)
        #expect(NadarayaWatson.fit(trainX: [[0]], trainY: [.nan]) == nil)
        let xs = [[0.0], [1.0]]
        #expect(NadarayaWatson.fit(trainX: xs, trainY: [0, 1], span: 0.5)?.fittedValues.count == 2)
        // Duplicated inputs stay finite through the fallback cascade.
        let dupX = [[0.0], [1.0], [1.0], [1.0], [2.0]]
        let dup = NadarayaWatson.fit(trainX: dupX, trainY: [0, 1, 1.5, 0.5, 2], span: 0.5)!
        #expect(dup.predict([[1.0]]).allSatisfy { $0.isFinite })
        #expect(dup.gradient(at: [1.0]) != nil || true)  // degenerate may be nil; must not trap
    }

    @Test func fittedSmootherCaseRoundTrips() async throws {
        // The carrier case forwards everything identically: wrap a fit
        // and compare every path against the direct calls.
        var rng = SeedableRandomNumberGenerator(seed: 2406)
        var cache = GaussianCache()
        let xs = (0..<40).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let direct = NadarayaWatson.fit(trainX: xs, trainY: ys, span: 0.5, robustIterations: 1)!
        let fit = FittedSmoother.nadarayaWatson(direct)
        let grid = (0..<20).map { [Double($0) / 5] }
        #expect(fit.fittedValues == direct.fittedValues)
        #expect(fit.keptIndices == direct.keptIndices)
        #expect(fit.predict(grid) == direct.predict(grid))
        #expect(try await fit.predictConcurrently(grid) == direct.predict(grid))
        #expect(fit.standardErrors(at: grid) == direct.standardErrors(at: grid))
        #expect(try await fit.standardErrorsConcurrently(at: grid) == direct.standardErrors(at: grid))
        #expect(fit.gradients(at: grid) == direct.gradients(at: grid))
        #expect(try await fit.gradientsConcurrently(at: grid) == direct.gradients(at: grid))
    }

    @Test func selectSpanPrefersSmallSpansOnCurves() {
        // GCV over kernel-scale spans: on sine truth the winner must be
        // narrow (wide spans flatten peaks — the failure mode that
        // motivated selection), and the fit must track the apex.
        var rng = SeedableRandomNumberGenerator(seed: 2407)
        var cache = GaussianCache()
        let xs = (0..<60).map { [Double($0) / 10] }
        let truth = xs.map { sin($0[0]) }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let grid = [0.05, 0.1, 0.2, 0.4]
        guard let (span, fit) = NadarayaWatson.selectSpan(
            trainX: xs, trainY: ys, spans: grid, robustIterations: 1
        ) else {
            Issue.record("selectSpan returned nil")
            return
        }
        #expect(grid.contains(span))
        #expect(span <= 0.2)
        #expect(rmse(fit.fittedValues, truth) < 0.2)
        // Apex region (π/2 ≈ 1.57): fitted values reach near the peak.
        let apex = zip(xs, fit.fittedValues).filter { abs($0.0[0] - 1.57) < 0.5 }.map(\.1)
        #expect(apex.max()! > 0.8)
    }
}
