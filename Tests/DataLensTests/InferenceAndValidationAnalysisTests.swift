import Foundation
import Testing
@testable import DataLens

@Suite("Inference and validation analysis")
struct InferenceAndValidationAnalysisTests {
    private func binomialFixture(_ count: Int = 140) -> ([[Double]], [Double]) {
        var rng = SeedableRandomNumberGenerator(seed: 0x1AF3_2026)
        let x = (0..<count).map { [Double($0) / Double(count - 1) * 2 - 1] }
        let y = x.map { row -> Double in
            let eta = -0.2 + 1.1 * row[0] - 0.35 * row[0] * row[0]
            return rng.nextBool(probability: 1 / (1 + exp(-eta))) ? 1 : 0
        }
        return (x, y)
    }

    @Test func likelihoodInferenceUsesPenalizedEDFAndFiniteConditionalIntervals() throws {
        let (x, y) = binomialFixture()
        let fit = try #require(LikelihoodAdditiveModel.fit(
            trainX: x, trainY: y, family: .binomial,
            specification: .init(defaultKnotCount: 2, penaltyWeight: 0.2,
                                 maxIterations: 100, tolerance: 1e-8)
        ).model)

        #expect(fit.inference.covarianceMethod == .penalizedObservedInformation)
        #expect(fit.inference.effectiveDegreesOfFreedom >= 1)
        #expect(fit.inference.effectiveDegreesOfFreedom < Double(fit.inference.coefficientCovariance.count))
        #expect(fit.diagnostics.effectiveDegreesOfFreedom == fit.inference.effectiveDegreesOfFreedom)
        for row in fit.inference.coefficientCovariance.indices {
            #expect(fit.inference.coefficientCovariance[row][row] >= 0)
            for column in 0..<row {
                #expect(abs(
                    fit.inference.coefficientCovariance[row][column]
                        - fit.inference.coefficientCovariance[column][row]
                ) < 1e-10)
            }
        }
        #expect(try #require(fit.linkStandardError(at: [0])) > 0)
        #expect(try #require(fit.standardError(at: [0])) > 0)
        let interval = try #require(fit.meanConfidenceInterval(at: [0]))
        #expect(interval.lowerBound > 0 && interval.upperBound < 1)
        #expect(interval.lowerBound <= interval.estimate && interval.estimate <= interval.upperBound)
        #expect(fit.meanConfidenceInterval(at: [0], confidenceLevel: -0.5) == nil)
        let partial = try #require(fit.partialEffectInterval(forPredictor: 0, count: 19))
        #expect(partial.x.count == 19)
        #expect(zip(partial.lowerBounds, partial.upperBounds).allSatisfy { $0.0 <= $0.1 })

        let unified = try #require(FittedStatisticalModel.fit(
            trainX: x, trainY: y,
            specification: .init(
                strategy: .additiveBinomial,
                likelihoodAdditive: .init(defaultKnotCount: 2, penaltyWeight: 0.2,
                                          maxIterations: 100, tolerance: 1e-8)
            )
        ))
        #expect(unified.standardError(at: [0]) != nil)
        #expect(unified.meanConfidenceInterval(at: [0]) != nil)
        #expect(unified.partialEffectInterval(forPredictor: 0, count: 11) != nil)
    }

    @Test func calibrationAndComparisonRequireMatchedOutOfFoldEvidence() throws {
        let (x, y) = binomialFixture()
        let specification = StatisticalModelSpecification(
            strategy: .additiveBinomial,
            likelihoodAdditive: .init(defaultKnotCount: 2, penaltyWeight: 0.2,
                                      maxIterations: 100, tolerance: 1e-8)
        )
        let first = try #require(CrossValidation.evaluate(
            trainX: x, trainY: y,
            configuration: .init(foldCount: 5, partitioning: .stratifiedBinary, seed: 9,
                                 specification: specification)
        ))
        let second = try #require(CrossValidation.evaluate(
            trainX: x, trainY: y,
            configuration: .init(foldCount: 5, partitioning: .stratifiedBinary, seed: 9,
                                 specification: specification)
        ))
        let calibration = try #require(first.binomialCalibration(binCount: 7))
        #expect(calibration.bins.count == 7)
        #expect(calibration.bins.map(\.observationCount).reduce(0, +) == x.count)
        #expect(calibration.brierScore.isFinite && calibration.expectedCalibrationError >= 0)

        let comparison = ModelComparison.compare(baseline: first, candidate: second)
        #expect(comparison.status == .comparable)
        #expect(comparison.pairedLosses.count == x.count)
        #expect(comparison.meanLossDifference == 0)
        #expect(comparison.tieCount == x.count)

        let remapped = try #require(CrossValidation.evaluate(
            trainX: x, trainY: y,
            configuration: .init(foldCount: 5, partitioning: .stratifiedBinary, seed: 10,
                                 specification: specification)
        ))
        #expect(ModelComparison.compare(baseline: first, candidate: remapped).status == .foldAssignmentMismatch)
    }

    @Test func bootstrapReportsSeededPredictionStabilityAndFailureThresholds() {
        let x = (0..<30).map { [Double($0) / 5] }
        let y = x.enumerated().map { 0.5 + 1.25 * $0.element[0] + Double($0.offset % 3 - 1) * 0.08 }
        let configuration = BootstrapConfiguration(
            replicateCount: 24, minimumSuccessFraction: 0.8, confidenceLevel: 0.9, seed: 42,
            specification: .init(degree: 1, spans: [0.8], robustIterations: 0, adaptiveContender: false)
        )
        let first = ModelResampling.bootstrap(
            trainX: x, trainY: y, queryPoints: [[1], [3]], configuration: configuration
        )
        let second = ModelResampling.bootstrap(
            trainX: x, trainY: y, queryPoints: [[1], [3]], configuration: configuration
        )
        #expect(first == second)
        #expect(first.status == .completed)
        #expect(first.successfulReplicates + first.failedReplicates == configuration.replicateCount)
        #expect(first.predictions.count == 2)
        #expect(first.predictions.allSatisfy {
            $0.interval.lowerBound.isFinite && $0.interval.upperBound.isFinite
                && $0.interval.lowerBound <= $0.interval.upperBound
        })

        let invalid = ModelResampling.bootstrap(
            trainX: x, trainY: y, queryPoints: [[1, 2]], configuration: configuration
        )
        #expect(invalid.status == .invalidInput)
        #expect(invalid.predictions.isEmpty)
    }
}
