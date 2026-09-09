import Foundation
import Testing
@testable import DataLens

/// Port of Numerical-Statistics' `LoessTests` (XCTest → swift-testing;
/// assertions mapped 1:1, same seeds, same tolerances).
@Suite("Loess")
struct LoessTests {
    @Test func weights() {
        #expect(abs(LoessWeight.tricube(0) - 1.0) <= 1e-12)
        #expect(abs(LoessWeight.tricube(0.5) - pow(1 - 0.125, 3)) <= 1e-12)
        #expect(abs(LoessWeight.tricube(1.0) - 0.0) <= 1e-12)
        #expect(abs(LoessWeight.tricube(2.0) - 0.0) <= 1e-12)
        #expect(abs(LoessWeight.bisquare(0) - 1.0) <= 1e-12)
    }

    @Test func reproducesLinear() {
        // Degree 1 reproduces lines exactly at any span.
        let xs = (0..<15).map { [Double($0)] }
        let ys = xs.map { 2 * $0[0] - 1 }
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 1)!
        for (f, y) in zip(fit.fittedValues, ys) {
            #expect(abs(f - y) <= 1e-9)
        }
        #expect(abs(fit.predict([7.5]) - 14.0) <= 1e-9)
    }

    @Test func reproducesQuadratic() {
        let xs = (0..<20).map { [Double($0) / 4] }
        let ys = xs.map { $0[0] * $0[0] }
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.6, degree: 2)!
        for (f, y) in zip(fit.fittedValues, ys) {
            #expect(abs(f - y) <= 1e-7)
        }
    }

    @Test func reproducesPlane() {
        // Multivariate degree 1 reproduces planes exactly.
        var rng = SeedableRandomNumberGenerator(seed: 2201)
        let xs = (0..<40).map { _ in [Double.random(in: 0...3, using: &rng),
                                       Double.random(in: 0...3, using: &rng)] }
        let ys = xs.map { 1 + 2 * $0[0] - $0[1] }
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.6, degree: 1)!
        for (f, y) in zip(fit.fittedValues, ys) {
            #expect(abs(f - y) <= 1e-7)
        }
    }

    @Test func robustToOutlier() {
        var ys = (0..<20).map { 2 * Double($0) }
        ys[10] = 500
        let xs = (0..<20).map { [Double($0)] }
        let robust = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 1)!
        let plain = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 1,
                              robustIterations: 0)!
        // Robust fit stays near the line at the outlier; plain fit is dragged.
        #expect(abs(robust.fittedValues[10] - 20) < abs(plain.fittedValues[10] - 20))
        #expect(abs(robust.fittedValues[10] - 20.0) <= 3.0)
    }

    @Test func symmetry() {
        let xs = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        let ys = [0.0, 1.0, 0.0, 1.0, 0.0]
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.8, degree: 1)!
        #expect(abs(fit.predict([1.0]) - fit.predict([3.0])) <= 1e-9)
    }

    @Test func standardErrors() {
        var rng = SeedableRandomNumberGenerator(seed: 2202)
        var cache = GaussianCache()
        let xs = (0..<60).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.4, degree: 1)!
        #expect(fit.sigma > 0.05)
        #expect(fit.sigma < 0.2)
        #expect(fit.trace > 1)
        #expect(fit.trace < 60)
        for x in [[1.0], [3.0], [5.0]] {
            let se = fit.standardError(at: x)!
            #expect(se > 0)
            #expect(se < 0.5)
        }
    }

    @Test func spanSelection() {
        var rng = SeedableRandomNumberGenerator(seed: 2203)
        var cache = GaussianCache()
        let xs = (0..<50).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        let (span, fit) = Loess.selectSpan(trainX: xs, trainY: ys,
                                           spans: [0.2, 0.4, 0.6, 0.9], degree: 1)!
        #expect([0.2, 0.4, 0.6, 0.9].contains(span))
        // The selected smoother tracks the sine wave.
        let rmse = sqrt(zip(ys, fit.fittedValues).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / 50)
        #expect(rmse < 0.25)
    }

    @Test func invalidInput() {
        #expect(Loess.fit(trainX: [], trainY: []) == nil)
        #expect(Loess.fit(trainX: [[0]], trainY: [1, 2]) == nil)
        #expect(Loess.fit(trainX: [[0]], trainY: [1], span: 1.5) == nil)
    }
}
