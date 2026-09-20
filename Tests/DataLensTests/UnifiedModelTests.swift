import Foundation
import Testing
@testable import DataLens

@Suite("Unified models and validation")
struct UnifiedModelTests {
    private func continuousFixture(_ count: Int = 30) -> ([[Double]], [Double]) {
        let x = (0..<count).map { [Double($0) / 5] }
        return (x, x.map { 1.75 * $0[0] + 0.25 })
    }

    @Test func automaticFitProvidesUnifiedTrainingContract() throws {
        let (x, y) = continuousFixture()
        let specification = StatisticalModelSpecification(
            degree: 1, spans: [0.75], robustIterations: 0, adaptiveContender: false
        )
        let model = try #require(FittedStatisticalModel.fit(
            trainX: x, trainY: y, specification: specification
        ))

        #expect(model.kind == .smoother)
        #expect(model.specification == specification)
        #expect(model.trainingPredictors == x)
        #expect(model.trainingResponses == y)
        #expect(model.keptIndices == Array(x.indices))
        #expect(model.diagnostics.responseFamily == .gaussian)
        #expect(model.residuals(.raw).allSatisfy { abs($0) < 1e-8 })
        #expect(abs(model.predict([2.4]) - 4.45) < 1e-8)
        #expect(model.standardError(at: [2.4]) != nil)
    }

    @Test func additiveFitUsesTheSamePredictionAndDiagnosticsSurface() throws {
        var x: [[Double]] = []
        var y: [Double] = []
        for first in 0..<6 {
            for second in 0..<6 {
                x.append([Double(first), Double(second)])
                y.append(1 + 2 * Double(first) - 3 * Double(second))
            }
        }
        let additive = try #require(AdditiveModel.fit(
            trainX: x, trainY: y, defaultSpan: 0.9, defaultDegree: 1,
            maxIterations: 100, tolerance: 1e-9
        ))
        let model = FittedStatisticalModel(additive: additive)

        #expect(model.kind == .additiveGaussian)
        #expect(model.diagnostics.responseFamily == .gaussian)
        #expect(model.diagnostics.deviance ?? .infinity < 1e-8)
        #expect(abs(model.predict([2.5, 1.5]) - 1.5) < 1e-7)
        #expect(model.gradient(at: [2.5, 1.5])?.map { $0.rounded() } == [2, -3])
        #expect(model.standardError(at: [2.5, 1.5]) == nil)
    }

    @Test func shuffledValidationIsDeterministicAndOutOfFold() throws {
        let (x, y) = continuousFixture()
        let configuration = ValidationConfiguration(
            foldCount: 5, partitioning: .shuffled, seed: 42,
            specification: .init(degree: 1, spans: [0.75], robustIterations: 0,
                                 adaptiveContender: false)
        )
        let first = try #require(CrossValidation.evaluate(trainX: x, trainY: y, configuration: configuration))
        let second = try #require(CrossValidation.evaluate(trainX: x, trainY: y, configuration: configuration))

        #expect(first == second)
        #expect(first.responseFamily == .gaussian)
        #expect(first.predictions.map(\.id) == Array(x.indices))
        #expect(Set(first.predictions.map(\.fold)).count == 5)
        #expect(first.folds.map(\.observationCount).reduce(0, +) == x.count)
        #expect(try #require(first.rootMeanSquaredError) < 1e-8)
        #expect(try #require(first.meanAbsoluteError) < 1e-8)
        #expect(first.meanDeviance == nil)
        #expect(first.primaryScore == first.rootMeanSquaredError)
    }

    @Test func blockedAndStratifiedValidationHonorTheirStatisticalRoles() throws {
        let (x, y) = continuousFixture(20)
        let blocked = try #require(CrossValidation.evaluate(
            trainX: x, trainY: y,
            configuration: .init(
                foldCount: 4, partitioning: .blocked,
                specification: .init(degree: 1, spans: [0.75], robustIterations: 0,
                                     adaptiveContender: false)
            )
        ))
        #expect(blocked.predictions.filter { $0.fold == 0 }.map(\.id) == [0, 1, 2, 3, 4])

        let binaryX = (0..<40).map { [Double($0) / 10] }
        let binaryY = (0..<40).map { $0 < 20 ? 0.0 : 1.0 }
        let binary = try #require(CrossValidation.evaluate(
            trainX: binaryX, trainY: binaryY,
            configuration: .init(foldCount: 4, partitioning: .stratifiedBinary, seed: 19,
                                 specification: .init(degree: 1, spans: [0.75], robustIterations: 0))
        ))
        #expect(binary.responseFamily == .binomial)
        #expect(binary.meanDeviance?.isFinite == true)
        #expect(binary.rootMeanSquaredError == nil)
        #expect(binary.folds.allSatisfy { $0.observationCount == 10 })
        #expect(CrossValidation.evaluate(
            trainX: x, trainY: y,
            configuration: .init(foldCount: 4, partitioning: .stratifiedBinary)
        ) == nil)
    }
}
