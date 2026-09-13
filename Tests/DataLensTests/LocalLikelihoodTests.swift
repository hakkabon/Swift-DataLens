import Foundation
import Testing
@testable import DataLens

/// Tests for clean-room local likelihood: Gaussian consistency with `Loess`,
/// parameter recovery on synthetic truth per family, separation clamping,
/// and input validation.
@Suite("Local likelihood")
struct LocalLikelihoodTests {
    func rmse(_ fitted: [Double], _ truth: [Double]) -> Double {
        sqrt(zip(fitted, truth).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(truth.count))
    }

    /// Knuth's Poisson sampler (test-only; λ small here so it is fast).
    func poissonKnuth(lambda: Double, rng: inout SeedableRandomNumberGenerator) -> Double {
        precondition(lambda >= 0, "Poisson rate must be non-negative")
        if lambda == 0 { return 0 }
        let l = exp(-lambda)
        var k = 0
        var p = 1.0
        repeat {
            k += 1
            p *= Double.random(in: 0..<1, using: &rng)
        } while p > l
        return Double(k - 1)
    }

    @Test func gaussianMatchesLoess() {
        // Gaussian local likelihood without robustness is the same weighted
        // least squares Loess solves; Loess always applies one bisquare
        // round (even robustIterations: 0 updates the weights once), so this
        // checks family agreement loosely — exactness is pinned below.
        var rng = SeedableRandomNumberGenerator(seed: 5150)
        var cache = GaussianCache()
        let xs = (0..<30).map { [Double($0) / 5] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let ll = LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 2,
                                      family: .gaussian, span: 0.5)!
        let loess = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 2,
                              robustIterations: 0)!
        for (a, b) in zip(ll.fittedValues, loess.fittedValues) {
            #expect(abs(a - b) <= 0.05)
        }
        #expect(abs(ll.predict([2.5]) - loess.predict([2.5])) <= 0.05)
    }

    @Test func irlsFixedPoint() {
        // The Newton core, checked directly: Gaussian β satisfies the normal
        // equations; binomial β zeroes the score. Both to tight tolerance.
        let xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
        let rows = xs.map { Loess.basis([$0], center: [0.0], degree: 1) }
        let ones = [Double](repeating: 1, count: xs.count)
        let start = [0.0, 0.0]
        // Gaussian on an exact line: recovers [1, 2] via BᵀBβ = Bᵀy.
        let yg = xs.map { 1 + 2 * $0 }
        let g = LocalLikelihood.localIRLS(rows: rows, values: yg, locality: ones,
                                           family: .gaussian, startBeta: start)!
        #expect(abs(g.beta[0] - 1.0) <= 1e-9)
        #expect(abs(g.beta[1] - 2.0) <= 1e-9)
        var maxResid = 0.0
        for a in 0..<2 {
            var lhs = 0.0
            var rhs = 0.0
            for (row, y) in zip(rows, yg) {
                lhs += row[a] * zip(row, g.beta).reduce(0.0) { $0 + $1.0 * $1.1 }
                rhs += row[a] * y
            }
            maxResid = max(maxResid, abs(lhs - rhs))
        }
        #expect(maxResid <= 1e-9)
        // Binomial: score Bᵀ(y − μ) vanishes at the solution.
        let yb = [0.0, 0.0, 1.0, 1.0, 1.0]
        let b = LocalLikelihood.localIRLS(rows: rows, values: yb, locality: ones,
                                           family: .binomial, startBeta: start)!
        var maxScore = 0.0
        for a in 0..<2 {
            var s = 0.0
            for (row, y) in zip(rows, yb) {
                let eta = zip(row, b.beta).reduce(0.0) { $0 + $1.0 * $1.1 }
                let mu = LocalLikelihood.mean(linkEta: eta, family: .binomial)
                s += row[a] * (y - mu)
            }
            maxScore = max(maxScore, abs(s))
        }
        #expect(maxScore <= 1e-6)
    }

    @Test func binomialRecovery() {
        var rng = SeedableRandomNumberGenerator(seed: 5151)
        let xs = (0..<80).map { _ in [Double.random(in: -2...2, using: &rng)] }
        let truth = xs.map { 1 / (1 + exp(-(2 * $0[0] - 1))) }
        let ys = zip(xs, truth).map { Double.random(in: 0..<1, using: &rng) < $0.1 ? 1.0 : 0.0 }
        let fit = LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                       family: .binomial, span: 0.5)!
        #expect(rmse(fit.fittedValues, truth) < 0.12)
        #expect(fit.deviance <= fit.nullDeviance)
        #expect(fit.fittedValues.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    @Test func poissonRecovery() {
        var rng = SeedableRandomNumberGenerator(seed: 5152)
        let xs = (0..<70).map { [Double($0) / 70 * 3] }
        let truth = xs.map { exp(0.2 + 0.6 * $0[0]) }
        let ys = zip(xs, truth).map { poissonKnuth(lambda: $0.1, rng: &rng) }
        let fit = LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                       family: .poisson, span: 0.5)!
        #expect(rmse(fit.fittedValues, truth) < 0.8)
        #expect(fit.deviance <= fit.nullDeviance)
        #expect(fit.fittedValues.allSatisfy { $0 >= 0 })
    }

    @Test func separationClamp() {
        // Step truth: neighborhoods deep in each region are pure 0/1.
        let xs = (0..<40).map { [Double($0) / 10 - 2] }
        let ys = xs.map { $0[0] > 0 ? 1.0 : 0.0 }
        let fit = LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                       family: .binomial, span: 0.4)!
        #expect(fit.fittedValues.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
        #expect(fit.fittedValues.first! < 0.05)
        #expect(fit.fittedValues.last! > 0.95)
        #expect(fit.deviance.isFinite)
    }

    @Test func standardErrors() {
        var rng = SeedableRandomNumberGenerator(seed: 5153)
        let xs = (0..<60).map { _ in [Double.random(in: -2...2, using: &rng)] }
        let ys = xs.map { Double.random(in: 0..<1, using: &rng) < 1 / (1 + exp(-$0[0])) ? 1.0 : 0.0 }
        let fit = LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                       family: .binomial, span: 0.5)!
        for x in [[-1.0], [0.0], [1.0]] {
            let se = fit.standardError(at: x)!
            #expect(se.isFinite && se > 0 && se < 0.5)
        }
        #expect(fit.standardError(at: [0.0, 0.0]) == nil)
    }

    @Test func spanSelection() {
        var rng = SeedableRandomNumberGenerator(seed: 5154)
        let xs = (0..<60).map { _ in [Double.random(in: -2...2, using: &rng)] }
        let truth = xs.map { 1 / (1 + exp(-(1.5 * $0[0]))) }
        let ys = zip(xs, truth).map { Double.random(in: 0..<1, using: &rng) < $0.1 ? 1.0 : 0.0 }
        let (span, fit) = LocalLikelihood.selectSpan(trainX: xs, trainY: ys,
                                                     spans: [0.3, 0.5, 0.7],
                                                     degree: 1, family: .binomial)!
        #expect([0.3, 0.5, 0.7].contains(span))
        #expect(rmse(fit.fittedValues, truth) < 0.15)
    }

    @Test func invalidInput() {
        #expect(LocalLikelihood.fit(trainX: [], trainY: []) == nil)
        #expect(LocalLikelihood.fit(trainX: [[0]], trainY: [1, 2]) == nil)
        #expect(LocalLikelihood.fit(trainX: [[0]], trainY: [1], degree: 5) == nil)
        #expect(LocalLikelihood.fit(trainX: [[0]], trainY: [1], span: 0) == nil)
        #expect(LocalLikelihood.fit(trainX: [[0], [1]], trainY: [0, 0.5],
                                    family: .binomial) == nil)
        #expect(LocalLikelihood.fit(trainX: [[0], [1]], trainY: [1, -1],
                                    family: .poisson) == nil)
    }
}
