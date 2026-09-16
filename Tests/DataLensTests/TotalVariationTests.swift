import Foundation
import Testing
@testable import DataLens

/// Tests for total-variation denoising: λ = 0 reproduction, step
/// recovery, KKT optimality on every fit, batch parity, missing masks,
/// and validation.
///
/// The KKT check is the load-bearing test: for `w = y − ŷ` there must
/// exist duals `s` with `Dᵀs = w/λ`, `|s| ≤ 1`, and `sign(s) = sign(Dŷ)`
/// on jumps. `Dᵀ`'s bidiagonal structure makes `s` a forward recurrence
/// (`s[0] = −w[0]/λ`, `s[i] = s[i−1] − w[i]/λ`), so existence, bounds,
/// and the closing equation are all directly checkable — no reference
/// implementation needed. Any ADMM wrongness fails here loudly.
@Suite("Total variation")
struct TotalVariationTests {
    func rmse(_ fitted: [Double], _ truth: [Double]) -> Double {
        sqrt(zip(fitted, truth).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(truth.count))
    }

    /// KKT verification for a fit on clean data with `lambda > 0`:
    /// returns the worst dual bound violation, the closing-equation
    /// error, and sign mismatches on jumps above `jumpTol`. All must be
    /// ~0 at a true optimum. (At λ = 0 the dual system degenerates —
    /// there the RMSE check below is the proof.)
    func kkt(_ xs: [Double], _ ys: [Double], _ fit: TotalVariation,
             jumpTol: Double) -> (bound: Double, closing: Double, signs: Int) {
        let n = xs.count
        let lambda = fit.lambda
        precondition(lambda > 0, "KKT duals divide by lambda")
        // Work entirely in sorted order (the penalty's own order).
        let order = xs.indices.sorted {
            xs[$0] < xs[$1] || (xs[$0] == xs[$1] && $0 < $1)
        }
        let w = order.map { ys[$0] - fit.fittedValues[$0] }
        let fs = order.map { fit.fittedValues[$0] }
        var s = [Double](repeating: 0, count: n - 1)
        var bound = 0.0
        var signs = 0
        s[0] = -w[0] / lambda
        for i in 1..<(n - 1) {
            s[i] = s[i - 1] - w[i] / lambda
        }
        for i in 0..<(n - 1) {
            bound = max(bound, max(0, abs(s[i]) - 1))
            if abs(fs[i + 1] - fs[i]) > jumpTol, s[i] * (fs[i + 1] - fs[i]) < 0 {
                signs += 1
            }
        }
        let closing = abs(s[n - 2] - w[n - 1] / lambda)
        return (bound, closing, signs)
    }

    @Test func lambdaZeroReproduces() {
        var rng = SeedableRandomNumberGenerator(seed: 2601)
        let xs = (0..<30).map { [Double($0) / 10] }
        let ys = xs.map { _ in Double.random(in: -2...2, using: &rng) }
        let fit = TotalVariation.fit(trainX: xs, trainY: ys, lambda: 0)!
        #expect(rmse(fit.fittedValues, ys) <= 1e-6)
    }

    @Test func recoversSteps() {
        // Piecewise-constant truth: the fit must fuse into few segments
        // near the truth (the estimator's home ground).
        var rng = SeedableRandomNumberGenerator(seed: 2602)
        var cache = GaussianCache()
        let n = 90
        let xs = (0..<n).map { [Double($0) / 10] }
        let truth = xs.map { $0[0] < 3 ? 0.0 : ($0[0] < 6 ? 2.0 : 1.0) }
        let ys = xs.map { truth[Int(round($0[0] * 10))] + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fit = TotalVariation.fit(trainX: xs, trainY: ys, lambda: 0.5)!
        #expect(rmse(fit.fittedValues, truth) < 0.15)
        #expect(fit.trace < 12)
        let (bound, closing, signs) = kkt(xs.map({ $0[0] }), ys, fit, jumpTol: 1e-4)
        #expect(bound <= 1e-4 && closing <= 1e-4 && signs == 0)
    }

    @Test func kktHoldsOnSine() {
        // Smooth truth under an L1 penalty: optimum still satisfies KKT
        // exactly — the check is about the solver, not the model.
        var rng = SeedableRandomNumberGenerator(seed: 2603)
        var cache = GaussianCache()
        let xs = (0..<50).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fit = TotalVariation.fit(trainX: xs, trainY: ys, lambda: 0.3)!
        let (bound, closing, signs) = kkt(xs.map({ $0[0] }), ys, fit, jumpTol: 1e-4)
        #expect(bound <= 1e-4 && closing <= 1e-4 && signs == 0)
    }

    @Test func batchEqualsPointwiseAndConcurrent() async throws {
        var rng = SeedableRandomNumberGenerator(seed: 2604)
        var cache = GaussianCache()
        let xs = (0..<40).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fit = TotalVariation.fit(trainX: xs, trainY: ys, lambda: 0.3)!
        let grid = (0..<30).map { [Double($0) / 10 + 0.05] }
        let batch = fit.predict(grid)
        #expect(batch == grid.map { fit.predict($0) })
        #expect(try await fit.predictConcurrently(grid) == batch)
        #expect(try await fit.standardErrorsConcurrently(at: grid) == fit.standardErrors(at: grid))
        #expect(try await fit.gradientsConcurrently(at: grid) == fit.gradients(at: grid))
        #expect(fit.predict([[99.0]]).allSatisfy { $0.isNaN })
        #expect(fit.standardErrors(at: [[99.0]]) == [nil])
        #expect(fit.gradient(at: [1.0]) == [0.0])
    }

    @Test func droppingMissingKeepsIndices() {
        var rng = SeedableRandomNumberGenerator(seed: 2605)
        var cache = GaussianCache()
        let n = 40
        let xs = (0..<n).map { [Double($0) / 10] }
        var ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        ys[5] = .nan
        ys[30] = .nan
        let fit = TotalVariation.fit(trainX: xs, trainY: ys, lambda: 0.3,
                                      droppingMissing: true)!
        #expect(fit.keptIndices == (0..<n).filter { $0 != 5 && $0 != 30 })
        #expect(fit.fittedValues.count == n - 2)
        #expect(fit.fittedValues.allSatisfy { $0.isFinite })
    }

    @Test func invalidInput() {
        #expect(TotalVariation.fit(trainX: [], trainY: [], lambda: 1) == nil)
        #expect(TotalVariation.fit(trainX: [[0]], trainY: [1, 2], lambda: 1) == nil)
        #expect(TotalVariation.fit(trainX: [[0, 1]], trainY: [1], lambda: 1) == nil)
        #expect(TotalVariation.fit(trainX: [[0]], trainY: [1], lambda: -1) == nil)
        #expect(TotalVariation.fit(trainX: [[0]], trainY: [1], lambda: .nan) == nil)
        #expect(TotalVariation.fit(trainX: [[0]], trainY: [.nan], lambda: 1) == nil)
        #expect(TotalVariation.fit(trainX: [[0], [1]], trainY: [0, 1], lambda: 1)?.fittedValues.count == 2)
    }

    @Test func fittedSmootherCaseRoundTrips() async throws {
        // The carrier case forwards everything identically: wrap a fit
        // and compare every path against the direct calls.
        var rng = SeedableRandomNumberGenerator(seed: 2606)
        var cache = GaussianCache()
        let xs = (0..<40).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let direct = TotalVariation.fit(trainX: xs, trainY: ys, lambda: 0.3)!
        let fit = FittedSmoother.totalVariation(direct)
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
