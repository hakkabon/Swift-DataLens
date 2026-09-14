import Foundation
import Testing
@testable import DataLens

/// Derivatives: local-polynomial slopes, exact on lines/planes and tracking
/// cosine on sine truth.
@Suite("Derivatives")
struct DerivativeTests {
    @Test func loessLinearGradient() {
        let xs = (0..<15).map { [Double($0)] }
        let ys = xs.map { 2 * $0[0] - 1 }
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 1)!
        for x in [[0.0], [7.5], [14.0]] {
            #expect(abs(fit.gradient(at: x)![0] - 2.0) <= 1e-9)
        }
    }

    @Test func loessPlaneGradient() {
        var rng = SeedableRandomNumberGenerator(seed: 8101)
        let xs = (0..<40).map { _ in [Double.random(in: 0...3, using: &rng),
                                       Double.random(in: 0...3, using: &rng)] }
        let ys = xs.map { 1 + 2 * $0[0] - $0[1] }
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.6, degree: 1)!
        let g = fit.gradient(at: [1.5, 1.5])!
        #expect(abs(g[0] - 2.0) <= 1e-7)
        #expect(abs(g[1] + 1.0) <= 1e-7)
    }

    @Test func loessSineGradient() {
        var rng = SeedableRandomNumberGenerator(seed: 8102)
        var cache = GaussianCache()
        let xs = (0..<60).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.05 * cache.nextStandardNormal(using: &rng) }
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.4, degree: 2)!
        for x in [[1.0], [2.0], [4.0]] {
            #expect(abs(fit.gradient(at: x)![0] - cos(x[0])) < 0.15)
        }
        #expect(fit.gradient(at: [1.0, 2.0]) == nil)
    }

    @Test func loessDegreeZeroGradientIsNil() {
        let xs = (0..<10).map { [Double($0)] }
        let ys = xs.map { sin($0[0]) }
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 0)!
        #expect(fit.gradient(at: [4.5]) == nil)
    }

    @Test func adaptiveLinearGradient() {
        let xs = (0..<15).map { [Double($0)] }
        let ys = xs.map { 2 * $0[0] - 1 }
        let fit = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1)!
        #expect(abs(fit.gradient(at: [7.5])![0] - 2.0) <= 1e-9)
    }

    @Test func likelihoodLinearGradient() {
        let xs = (0..<20).map { [Double($0) / 2] }
        let ys = xs.map { 1 + 3 * $0[0] }
        let fit = LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                       family: .gaussian, span: 0.6)!
        let g = fit.gradient(at: [4.0])!
        #expect(abs(g[0] - 3.0) <= 1e-6)
    }

    @Test func gradientBatchesAgree() async throws {
        let xs = (0..<30).map { [Double($0) / 5] }
        let ys = xs.map { sin($0[0]) }
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 2)!
        let grid = (0..<10).map { [Double($0) / 2] }
        let batch = fit.gradients(at: grid)
        #expect(batch == grid.map { fit.gradient(at: $0) })
        let gradConc = try await fit.gradientsConcurrently(at: grid)
        #expect(gradConc == batch)
    }
}

/// Extrapolation policies: polynomial extends, nearest goes flat at the
/// edge fit, unavailable refuses — on values and standard errors.
@Suite("Extrapolation")
struct ExtrapolationTests {
    func sineFit() -> Loess {
        let xs = (0..<30).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) }
        return Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 2)!
    }

    @Test func loessPolicies() {
        let fit = sineFit()
        let outside = [100.0]
        // Polynomial (default): finite extension of the local fit.
        #expect(fit.predict(outside).isFinite)
        #expect(fit.predict(outside) == fit.predict(outside, extrapolation: .polynomial))
        // Nearest: the edge fitted value (nearest training point is last).
        #expect(fit.predict(outside, extrapolation: .nearest) == fit.fittedValues.last!)
        // Unavailable: NaN / nil.
        #expect(fit.predict(outside, extrapolation: .unavailable).isNaN)
        #expect(fit.standardError(at: outside, extrapolation: .unavailable) == nil)
        // Nearest SE comes from the edge point and is finite.
        let se = fit.standardError(at: outside, extrapolation: .nearest)!
        #expect(se.isFinite && se >= 0)
        // Inside the hull, policies agree with the default.
        let inside = [1.5]
        #expect(fit.predict(inside, extrapolation: .nearest) == fit.predict(inside))
        #expect(fit.predict(inside, extrapolation: .unavailable) == fit.predict(inside))
    }

    @Test func adaptivePolicies() {
        let xs = (0..<20).map { [Double($0) / 2] }
        let ys = xs.map { sin($0[0]) }
        let fit = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1)!
        #expect(fit.predict([100.0]).isFinite)
        #expect(fit.predict([100.0], extrapolation: .nearest) == fit.fittedValues.last!)
        #expect(fit.predict([100.0], extrapolation: .unavailable).isNaN)
        #expect(fit.standardError(at: [100.0], extrapolation: .unavailable) == nil)
    }

    @Test func likelihoodPolicies() {
        var rng = SeedableRandomNumberGenerator(seed: 8103)
        let xs = (0..<40).map { _ in [Double.random(in: -2...2, using: &rng)] }
        let ys = zip(xs, xs).map { Double.random(in: 0..<1, using: &rng) < 1 / (1 + exp(-$0.0[0])) ? 1.0 : 0.0 }
        let fit = LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                       family: .binomial, span: 0.5)!
        let outside = [10.0]
        let p = fit.predict(outside)
        #expect(p.isFinite && p >= 0 && p <= 1)
        let nearest = xs.enumerated().max(by: { $0.element[0] < $1.element[0] })!.offset
        #expect(fit.predict(outside, extrapolation: .nearest) == fit.fittedValues[nearest])
        #expect(fit.predict(outside, extrapolation: .unavailable).isNaN)
    }

    @Test func batchPolicies() async throws {
        let fit = sineFit()
        let grid = [[1.0], [100.0]]
        #expect(fit.predict(grid, extrapolation: .unavailable).map { $0.isNaN } == [false, true])
        let policyConc = try await fit.predictConcurrently(grid, extrapolation: .nearest)
        #expect(policyConc == fit.predict(grid, extrapolation: .nearest))
        #expect(fit.standardErrors(at: grid, extrapolation: .unavailable).map { $0 == nil }
            == [false, true])
    }
}

/// Missing data: dropping rows matches the clean fit exactly and reports
/// the mask; without the flag such rows fail validation.
@Suite("Missing data")
struct MissingDataTests {
    func cleanSine() -> (xs: [[Double]], ys: [Double]) {
        let xs = (0..<30).map { [Double($0) / 5] }
        return (xs, xs.map { sin($0[0]) })
    }

    func rmseBelow(_ fitted: [Double], _ truth: [Double], _ bound: Double) -> Bool {
        sqrt(zip(fitted, truth).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(truth.count)) < bound
    }

    @Test func loessDropMatchesClean() {
        let (xs, ys) = cleanSine()
        // Appended NaN rows vanish before fitting: bit-identical results.
        let dirtyX = xs + [[.nan], [1.0], [.infinity]]
        let dirtyY = ys + [.nan, .infinity, 0.0]
        let dropped = Loess.fit(trainX: dirtyX, trainY: dirtyY, span: 0.5, degree: 2,
                                droppingMissing: true)!
        let clean = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 2)!
        #expect(dropped.fittedValues == clean.fittedValues)
        #expect(dropped.keptIndices == Array(0..<30))
        #expect(dropped.trace == clean.trace)
        // Interior drops change neighborhoods slightly: count, mask, and a
        // sane recovery (not exactness).
        var dirtyX2 = xs
        var dirtyY2 = ys
        dirtyX2[7] = [.nan]
        dirtyY2[13] = .infinity
        let partial = Loess.fit(trainX: dirtyX2, trainY: dirtyY2, span: 0.5, degree: 2,
                                droppingMissing: true)!
        #expect(partial.fittedValues.count == 28)
        #expect(partial.keptIndices == Array(0..<30).filter { $0 != 7 && $0 != 13 })
        let keptTruth = partial.keptIndices.map { sin(xs[$0][0]) }
        #expect(rmseBelow(partial.fittedValues, keptTruth, 0.2))
        // Untouched rows validate strictly.
        #expect(Loess.fit(trainX: dirtyX, trainY: dirtyY, span: 0.5, degree: 2) == nil)
    }

    @Test func loessDropAllMissingIsNil() {
        let xs = [[0.0], [1.0]]
        #expect(Loess.fit(trainX: xs, trainY: [.nan, .nan], droppingMissing: true) == nil)
    }

    @Test func adaptiveDropMatchesClean() {
        let (xs, ys) = cleanSine()
        let dirtyY = ys + [.nan, .infinity]
        let dirtyX = xs + [[0.0], [1.0]]
        let dropped = AdaptiveLoess.fit(trainX: dirtyX, trainY: dirtyY, degree: 1,
                                        droppingMissing: true)!
        let clean = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1)!
        #expect(dropped.fittedValues == clean.fittedValues)
        #expect(dropped.selectedNeighborhoods == clean.selectedNeighborhoods)
        #expect(dropped.keptIndices == Array(0..<30))
    }

    @Test func likelihoodDropMatchesClean() {
        var rng = SeedableRandomNumberGenerator(seed: 8104)
        let xs = (0..<40).map { _ in [Double.random(in: -2...2, using: &rng)] }
        let ys = zip(xs, xs).map { Double.random(in: 0..<1, using: &rng) < 1 / (1 + exp(-$0.0[0])) ? 1.0 : 0.0 }
        let dirtyY = ys + [.nan]
        let dirtyX = xs + [[0.0]]
        let dropped = LocalLikelihood.fit(trainX: dirtyX, trainY: dirtyY, degree: 1,
                                           family: .binomial, span: 0.5,
                                           droppingMissing: true)!
        let clean = LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                         family: .binomial, span: 0.5)!
        #expect(dropped.fittedValues == clean.fittedValues)
        #expect(dropped.keptIndices == Array(0..<40))
    }

    @Test func automaticDropRoutes() {
        var rng = SeedableRandomNumberGenerator(seed: 8105)
        let xs = (0..<40).map { _ in [Double.random(in: -2...2, using: &rng)] }
        var ys = zip(xs, xs).map { Double.random(in: 0..<1, using: &rng) < 1 / (1 + exp(-$0.0[0])) ? 1.0 : 0.0 }
        ys[3] = .nan
        let (fit, summary) = AutomaticSmoother.fit(trainX: xs, trainY: ys, droppingMissing: true)!
        guard case .likelihood(let ll) = fit else {
            Issue.record("expected likelihood fit, got \(fit)")
            return
        }
        #expect(ll.family == .binomial)
        #expect(fit.keptIndices == Array(0..<40).filter { $0 != 3 })
        #expect(summary.notes.isEmpty)
        #expect(AutomaticSmoother.fit(trainX: xs, trainY: ys) == nil)
    }
}
