import Foundation
import Testing
@testable import DataLens

/// Batch and concurrent evaluation: shared-index batches equal pointwise
/// calls, and concurrent variants are bit-identical to sequential ones
/// (same per-point arithmetic, index-ordered reductions).
@Suite("Batch evaluation")
struct BatchTests {
    func sineData(n: Int, seed: UInt64) -> (xs: [[Double]], ys: [Double]) {
        var rng = SeedableRandomNumberGenerator(seed: seed)
        var cache = GaussianCache()
        let xs = (0..<n).map { [Double($0) / 10] }
        let ys = xs.map { sin($0[0]) + 0.1 * cache.nextStandardNormal(using: &rng) }
        return (xs, ys)
    }

    @Test func loessBatchEqualsPointwise() async {
        let (xs, ys) = sineData(n: 30, seed: 601)
        let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 2)!
        let grid = (0..<20).map { [Double($0) / 4] }
        let batch = fit.predict(grid)
        #expect(batch == grid.map { fit.predict($0) })
        #expect(await fit.predictConcurrently(grid) == batch)
        let seBatch = fit.standardErrors(at: grid)
        #expect(seBatch.count == grid.count)
        for (i, x) in grid.enumerated() {
            #expect(seBatch[i] == fit.standardError(at: x))
        }
        #expect(await fit.standardErrorsConcurrently(at: grid) == seBatch)
        #expect(fit.predict([]) == [])
        #expect(await fit.predictConcurrently([]) == [])
        #expect(fit.predict([[0, 0]]).first!.isNaN)
        #expect(fit.standardErrors(at: [[0, 0]]) == [nil])
    }

    @Test func loessConcurrentFitEqualsSync() async {
        let (xs, ys) = sineData(n: 30, seed: 602)
        let sync = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 2)!
        let conc = await Loess.fitConcurrently(trainX: xs, trainY: ys, span: 0.5, degree: 2)!
        #expect(conc.fittedValues == sync.fittedValues)
        #expect(conc.weights == sync.weights)
        #expect(conc.trace == sync.trace)
        #expect(conc.sigma == sync.sigma)
        #expect(await Loess.fitConcurrently(trainX: [], trainY: []) == nil)
    }

    @Test func adaptiveBatchEqualsPointwise() async {
        let (xs, ys) = sineData(n: 25, seed: 603)
        let fit = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1)!
        let grid = (0..<15).map { [Double($0) / 5] }
        let batch = fit.predict(grid)
        #expect(batch == grid.map { fit.predict($0) })
        #expect(await fit.predictConcurrently(grid) == batch)
        let seBatch = fit.standardErrors(at: grid)
        for (i, x) in grid.enumerated() {
            #expect(seBatch[i] == fit.standardError(at: x))
        }
        #expect(await fit.standardErrorsConcurrently(at: grid) == seBatch)
    }

    @Test func adaptiveConcurrentFitEqualsSync() async {
        let (xs, ys) = sineData(n: 25, seed: 604)
        let sync = AdaptiveLoess.fit(trainX: xs, trainY: ys, degree: 1)!
        let conc = await AdaptiveLoess.fitConcurrently(trainX: xs, trainY: ys, degree: 1)!
        #expect(conc.fittedValues == sync.fittedValues)
        #expect(conc.selectedNeighborhoods == sync.selectedNeighborhoods)
        #expect(conc.bandwidths == sync.bandwidths)
        #expect(conc.weights == sync.weights)
        #expect(conc.trace == sync.trace)
        #expect(conc.sigma == sync.sigma)
    }

    @Test func likelihoodBatchEqualsPointwise() async {
        var rng = SeedableRandomNumberGenerator(seed: 605)
        let xs = (0..<40).map { _ in [Double.random(in: -2...2, using: &rng)] }
        let ys = zip(xs, xs).map { Double.random(in: 0..<1, using: &rng) < 1 / (1 + exp(-$0.0[0])) ? 1.0 : 0.0 }
        let fit = LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                       family: .binomial, span: 0.5)!
        let grid = (0..<15).map { [Double($0) / 4 - 2] }
        let batch = fit.predict(grid)
        #expect(batch == grid.map { fit.predict($0) })
        #expect(await fit.predictConcurrently(grid) == batch)
        let seBatch = fit.standardErrors(at: grid)
        for (i, x) in grid.enumerated() {
            #expect(seBatch[i] == fit.standardError(at: x))
        }
        #expect(await fit.standardErrorsConcurrently(at: grid) == seBatch)
        #expect(fit.predict([]) == [])
    }

    @Test func likelihoodConcurrentFitEqualsSync() async {
        var rng = SeedableRandomNumberGenerator(seed: 606)
        let xs = (0..<40).map { _ in [Double.random(in: -2...2, using: &rng)] }
        let ys = zip(xs, xs).map { Double.random(in: 0..<1, using: &rng) < 1 / (1 + exp(-$0.0[0])) ? 1.0 : 0.0 }
        let sync = LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                        family: .binomial, span: 0.5)!
        let conc = await LocalLikelihood.fitConcurrently(trainX: xs, trainY: ys, degree: 1,
                                                          family: .binomial, span: 0.5)!
        #expect(conc.fittedValues == sync.fittedValues)
        #expect(conc.linearPredictors == sync.linearPredictors)
        #expect(conc.trace == sync.trace)
        #expect(conc.deviance == sync.deviance)
        #expect(conc.sigma == sync.sigma)
    }
}
