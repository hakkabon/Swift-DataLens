import Foundation
import Testing
@testable import DataLens

@Suite("Likelihood additive model")
struct LikelihoodAdditiveModelTests {
    private func binomialFixture(_ count: Int = 160) -> ([[Double]], [Double]) {
        var rng = SeedableRandomNumberGenerator(seed: 0xB1A0_2026)
        let x = (0..<count).map { index in [Double(index) / Double(count - 1) * 2 - 1] }
        let y = x.map { row -> Double in
            let eta = -0.35 + 1.15 * row[0] - 0.45 * row[0] * row[0]
            let probability = 1 / (1 + exp(-eta))
            return rng.nextBool(probability: probability) ? 1 : 0
        }
        return (x, y)
    }

    private func poissonFixture(_ count: Int = 180) -> ([[Double]], [Double]) {
        var rng = SeedableRandomNumberGenerator(seed: 0xA015_2026)
        let x = (0..<count).map { index in [Double(index) / Double(count - 1) * 2 - 1] }
        let y = x.map { row -> Double in
            let mean = exp(0.35 + 0.55 * row[0] - 0.3 * row[0] * row[0])
            return Double(poisson(mean: mean, using: &rng))
        }
        return (x, y)
    }

    @Test func binomialIRLSConvergesAndImprovesOnTheNullModel() throws {
        let (x, y) = binomialFixture()
        let result = LikelihoodAdditiveModel.fit(
            trainX: x, trainY: y, family: .binomial,
            specification: .init(defaultKnotCount: 2, penaltyWeight: 0.05,
                                 maxIterations: 100, tolerance: 1e-8)
        )
        let fit = try #require(result.model)

        #expect(result.status == .converged)
        #expect(result.iterations <= 100)
        #expect(fit.deviance < fit.nullDeviance)
        #expect(fit.scoreInfinityNorm < 1e-5)
        #expect(fit.fittedValues.allSatisfy { $0 > 0 && $0 < 1 })
        #expect(fit.predict([-0.8]) < fit.predict([0.5]))
        #expect(fit.residuals(.deviance).allSatisfy { $0.isFinite })
        #expect(fit.partialEffect(forPredictor: 0, count: 23)?.effect.count == 23)
    }

    @Test func poissonIRLSConvergesAndReportsFiniteFamilyDiagnostics() throws {
        let (x, y) = poissonFixture()
        let result = LikelihoodAdditiveModel.fit(
            trainX: x, trainY: y, family: .poisson,
            specification: .init(defaultKnotCount: 2, penaltyWeight: 0.1,
                                 maxIterations: 100, tolerance: 1e-8)
        )
        let fit = try #require(result.model)

        #expect(result.status == .converged)
        #expect(fit.deviance < fit.nullDeviance)
        #expect(fit.fittedValues.allSatisfy { $0.isFinite && $0 > 0 })
        #expect(fit.diagnostics.responseFamily == .poisson)
        #expect(fit.diagnostics.linkFunction == .log)
        #expect(fit.gradient(at: [0])?.first?.isFinite == true)
    }

    @Test func fitStatusSeparatesInvalidAndUnconvergedRequests() {
        let invalid = LikelihoodAdditiveModel.fit(
            trainX: [[0], [1]], trainY: [0, 0.5], family: .binomial
        )
        #expect(invalid.status == .invalidInput)
        #expect(invalid.model == nil)

        let (x, y) = binomialFixture()
        let incomplete = LikelihoodAdditiveModel.fit(
            trainX: x, trainY: y, family: .binomial,
            specification: .init(defaultKnotCount: 2, penaltyWeight: 0.05,
                                 maxIterations: 1, tolerance: 1e-14)
        )
        #expect(incomplete.status == .iterationLimit)
        #expect(incomplete.model == nil)
    }

    @Test func unifiedLikelihoodSpecificationsRouteToTheirDeclaredFamilies() throws {
        let (x, y) = binomialFixture(100)
        let specification = StatisticalModelSpecification(
            strategy: .additiveBinomial,
            likelihoodAdditive: .init(defaultKnotCount: 2, penaltyWeight: 0.1,
                                      maxIterations: 100, tolerance: 1e-8)
        )
        let encoded = try JSONEncoder().encode(specification)
        #expect(try JSONDecoder().decode(StatisticalModelSpecification.self, from: encoded) == specification)
        let model = try #require(FittedStatisticalModel.fit(
            trainX: x, trainY: y, specification: specification
        ))
        #expect(model.kind == .additiveBinomial)
        #expect(model.diagnostics.responseFamily == .binomial)
        #expect((model.diagnostics.deviance ?? .infinity) < (model.diagnostics.nullDeviance ?? -.infinity))
        #expect(model.standardError(at: [0]) != nil)
        #expect(model.meanConfidenceInterval(at: [0]) != nil)

        let validation = try #require(CrossValidation.evaluate(
            trainX: x, trainY: y,
            configuration: .init(foldCount: 4, partitioning: .stratifiedBinary, seed: 17,
                                 specification: specification)
        ))
        #expect(validation.responseFamily == .binomial)
        #expect(validation.meanDeviance?.isFinite == true)
        #expect(validation.rootMeanSquaredError == nil)

        let (poissonX, poissonY) = poissonFixture(100)
        let poissonSpecification = StatisticalModelSpecification(
            strategy: .additivePoisson,
            likelihoodAdditive: .init(defaultKnotCount: 2, penaltyWeight: 0.1,
                                      maxIterations: 100, tolerance: 1e-8)
        )
        let poissonModel = try #require(FittedStatisticalModel.fit(
            trainX: poissonX, trainY: poissonY, specification: poissonSpecification
        ))
        #expect(poissonModel.kind == .additivePoisson)
        #expect(poissonModel.diagnostics.responseFamily == .poisson)
        #expect(poissonModel.predict([0]) > 0)
    }

    private func poisson<R: RandomNumberGenerator>(mean: Double, using rng: inout R) -> Int {
        let threshold = exp(-mean)
        var product = 1.0
        var count = 0
        repeat {
            count += 1
            product *= Double.random(in: 0..<1, using: &rng)
        } while product > threshold
        return count - 1
    }
}
