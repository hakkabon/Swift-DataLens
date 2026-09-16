import Foundation
import Testing
@testable import DataLens

/// Tests for Whittaker–Eilers penalized smoothing: interpolation at
/// λ = 0, exact lines under an order-2 penalty, GCV selection, batch
/// parity, order restoration on shuffled input, segment-slope gradients,
/// missing-data masks, and validation.
@Suite("Whittaker-Eilers")
struct WhittakerEilersTests {
    func rmse(_ fitted: [Double], _ truth: [Double]) -> Double {
        sqrt(zip(fitted, truth).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(truth.count))
    }

    @Test func lambdaZeroInterpolates() {
        // λ = 0 makes M = I: fitted values are the responses, bit-identically.
        var rng = SeedableRandomNumberGenerator(seed: 2501)
        let xs = (0..<30).map { [Double($0) / 10] }
        let ys = xs.map { _ in Double.random(in: -2...2, using: &rng) }
        let fit = WhittakerEilers.fit(trainX: xs, trainY: ys, lambda: 0)!
        #expect(fit.fittedValues == ys)
        #expect(fit.trace == Double(xs.count))
    }

    @Test func recoversLinesUnderOrderTwo() {
        // Second differences of a line vanish: small λ reproduces it.
        let xs = (0..<40).map { [Double($0) / 10] }
        let ys = xs.map { 2 * $0[0] - 1 }
        let fit = WhittakerEilers.fit(trainX: xs, trainY: ys, lambda: 1, order: 2)!
        #expect(rmse(fit.fittedValues, ys) <= 1e-9)
    }

    @Test func selectLambdaRecoversSine() {
        var rng = SeedableRandomNumberGenerator(seed: 2502)
        var cache = GaussianCache()
        let xs = (0..<60).map { [Double($0) / 10] }
        let truth = xs.map { sin($0[0]) }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let grid = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        guard let (lambda, fit) = WhittakerEilers.selectLambda(
            trainX: xs, trainY: ys, lambdas: grid, order: 2
        ) else {
            Issue.record("selectLambda returned nil")
            return
        }
        #expect(grid.contains(lambda))
        #expect(rmse(fit.fittedValues, truth) < 0.2)
        #expect(fit.trace > 1 && fit.trace < 60)
        #expect(fit.sigma > 0.05 && fit.sigma < 0.3)
        #expect(fit.standardErrors.allSatisfy { $0 > 0 && $0.isFinite })
    }

    @Test func batchEqualsPointwiseAndConcurrent() async throws {
        var rng = SeedableRandomNumberGenerator(seed: 2503)
        var cache = GaussianCache()
        let xs = (0..<40).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fit = WhittakerEilers.fit(trainX: xs, trainY: ys, lambda: 100, order: 2)!
        let grid = (0..<30).map { [Double($0) / 10 + 0.05] }
        let batch = fit.predict(grid)
        #expect(batch == grid.map { fit.predict($0) })
        #expect(try await fit.predictConcurrently(grid) == batch)
        let seBatch = fit.standardErrors(at: grid)
        #expect(try await fit.standardErrorsConcurrently(at: grid) == seBatch)
        #expect(seBatch.allSatisfy { ($0 ?? -1) > 0 })
        #expect(try await fit.gradientsConcurrently(at: grid) == fit.gradients(at: grid))
        // Outside the hull: gaps, not inventions.
        #expect(fit.predict([[-1.0]]).allSatisfy { $0.isNaN })
        #expect(fit.standardErrors(at: [[99.0]]) == [nil])
        #expect(fit.gradient(at: [99.0]) == nil)
    }

    @Test func shuffledInputRestoresOrder() {
        // Rows in reverse: fitted values must align to input order, and
        // the fit must equal the sorted-input fit reordered.
        var rng = SeedableRandomNumberGenerator(seed: 2504)
        var cache = GaussianCache()
        let n = 40
        let xs = (0..<n).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fwd = WhittakerEilers.fit(trainX: xs, trainY: ys, lambda: 100)!
        let revX = xs.reversed().map { $0 }
        let revY = ys.reversed().map { $0 }
        let rev = WhittakerEilers.fit(trainX: revX, trainY: revY, lambda: 100)!
        for i in 0..<n {
            #expect(abs(rev.fittedValues[n - 1 - i] - fwd.fittedValues[i]) <= 1e-9)
        }
        #expect(rev.keptIndices == Array(0..<n))
    }

    @Test func gradientsAreSegmentSlopes() {
        // Piecewise-linear interpolant: central differences away from
        // knots equal the segment slope to solver noise.
        let xs = (0..<20).map { [Double($0)] }
        let ys = xs.map { sin($0[0]) }
        let fit = WhittakerEilers.fit(trainX: xs, trainY: ys, lambda: 10)!
        let h = 1e-7
        for x in [0.5, 3.5, 9.5, 15.5, 18.5].map({ [$0 + 0.13] }) {
            guard let g = fit.gradient(at: x) else {
                Issue.record("nil gradient at \(x)")
                continue
            }
            let numeric = (fit.predict([x[0] + h]) - fit.predict([x[0] - h])) / (2 * h)
            #expect(abs(g[0] - numeric) <= 1e-6)
            #expect(g.count == 1)
        }
    }

    @Test func droppingMissingKeepsIndices() {
        var rng = SeedableRandomNumberGenerator(seed: 2505)
        var cache = GaussianCache()
        let n = 40
        let xs = (0..<n).map { [Double($0) / 10] }
        var ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        ys[5] = .nan
        ys[30] = .nan
        let fit = WhittakerEilers.fit(trainX: xs, trainY: ys, lambda: 100,
                                      droppingMissing: true)!
        #expect(fit.keptIndices == (0..<n).filter { $0 != 5 && $0 != 30 })
        #expect(fit.fittedValues.count == n - 2)
        #expect(fit.fittedValues.allSatisfy { $0.isFinite })
        // Same data without the missing rows fits identically.
        let cleanX = xs.enumerated().filter { $0.offset != 5 && $0.offset != 30 }.map(\.element)
        let cleanY = ys.enumerated().filter { $0.offset != 5 && $0.offset != 30 }.map(\.element)
        let clean = WhittakerEilers.fit(trainX: cleanX, trainY: cleanY, lambda: 100)!
        #expect(fit.fittedValues == clean.fittedValues)
    }

    @Test func invalidInput() {
        #expect(WhittakerEilers.fit(trainX: [], trainY: [], lambda: 1) == nil)
        #expect(WhittakerEilers.fit(trainX: [[0]], trainY: [1, 2], lambda: 1) == nil)
        #expect(WhittakerEilers.fit(trainX: [[0, 1]], trainY: [1], lambda: 1) == nil)
        #expect(WhittakerEilers.fit(trainX: [[0]], trainY: [1], lambda: -1) == nil)
        #expect(WhittakerEilers.fit(trainX: [[0]], trainY: [1], lambda: .nan) == nil)
        #expect(WhittakerEilers.fit(trainX: [[0]], trainY: [1], lambda: 1, order: 0) == nil)
        #expect(WhittakerEilers.fit(trainX: [[0]], trainY: [1], lambda: 1, order: 4) == nil)
        #expect(WhittakerEilers.fit(trainX: [[0]], trainY: [.nan], lambda: 1) == nil)
        #expect(WhittakerEilers.fit(trainX: [[0], [1]], trainY: [0, 1], lambda: 1)?.fittedValues.count == 2)
    }

    @Test func fittedSmootherCaseRoundTrips() async throws {
        // The carrier case forwards everything identically: wrap a fit
        // and compare every path against the direct calls.
        var rng = SeedableRandomNumberGenerator(seed: 2506)
        var cache = GaussianCache()
        let xs = (0..<40).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let direct = WhittakerEilers.fit(trainX: xs, trainY: ys, lambda: 100)!
        let fit = FittedSmoother.whittakerEilers(direct)
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
}
