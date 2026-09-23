import Foundation
import Testing
@testable import DataLens

@Suite("Multivariate statistics")
struct MultivariateModelTests {
    private func tensorFixture() -> ([[Double]], [Double]) {
        var x: [[Double]] = []
        var y: [Double] = []
        for first in 0..<9 {
            for second in 0..<9 {
                let a = -1 + Double(first) / 4
                let b = -1 + Double(second) / 4
                x.append([a, b])
                y.append(1 + 0.7 * a - 0.4 * b + 1.2 * a * b)
            }
        }
        return (x, y)
    }

    private var tensorTerms: [MultivariateTermSpecification] {
        [
            .spline(.init(predictorIndex: 0, knotCount: 0)),
            .spline(.init(predictorIndex: 1, knotCount: 0)),
            .tensorProduct(.init(
                firstPredictorIndex: 0, secondPredictorIndex: 1,
                firstKnotCount: 0, secondKnotCount: 0
            )),
        ]
    }

    @Test func tensorInteractionRecoversSurfaceAndProducesContours() throws {
        let (x, y) = tensorFixture()
        let fit = try #require(MultivariateModel.fit(
            trainX: x, trainY: y, family: .gaussian,
            specification: .init(terms: tensorTerms, penaltyWeight: 1e-6,
                                 maxIterations: 20, tolerance: 1e-9)
        ).model)
        let point = [0.3, -0.4]
        let truth = 1 + 0.7 * point[0] - 0.4 * point[1] + 1.2 * point[0] * point[1]
        #expect(abs(fit.predict(point) - truth) < 1e-5)
        let gradient = try #require(fit.gradient(at: point))
        #expect(abs(gradient[0] - (0.7 + 1.2 * point[1])) < 2e-4)
        #expect(abs(gradient[1] - (-0.4 + 1.2 * point[0])) < 5e-5)
        #expect(fit.inference.effectiveDegreesOfFreedom < 16)
        let contour = try #require(fit.contour(
            xPredictorIndex: 0, yPredictorIndex: 1, baseline: [0, 0], xCount: 21, yCount: 19
        ))
        #expect(contour.values.count == 19 && contour.values.allSatisfy { $0.count == 21 })
        #expect(!contour.segments(at: [1]).isEmpty)
    }

    @Test func categoricalTreatmentCodingIsExplicitAndRejectsUnknownLevels() throws {
        var x: [[Double]] = []
        var y: [Double] = []
        for code in [10.0, 20.0, 30.0] {
            for index in 0..<12 {
                let value = -1 + Double(index) / 5.5
                x.append([value, code])
                let categoryEffect = code == 10 ? 0.0 : code == 20 ? -1.25 : 1.75
                y.append(2 + 0.5 * value + categoryEffect)
            }
        }
        let fit = try #require(MultivariateModel.fit(
            trainX: x, trainY: y, family: .gaussian,
            specification: .init(
                terms: [
                    .spline(.init(predictorIndex: 0, knotCount: 0)),
                    .categorical(.init(predictorIndex: 1, levels: [10, 20, 30], referenceLevel: 10)),
                ], penaltyWeight: 1e-6, maxIterations: 20, tolerance: 1e-9
            )
        ).model)
        #expect(abs((fit.predict([0, 30]) - fit.predict([0, 10])) - 1.75) < 1e-5)
        #expect(abs((fit.predict([0, 20]) - fit.predict([0, 10])) + 1.25) < 1e-5)
        #expect(fit.predict([0, 99]).isNaN)
        #expect(fit.gradient(at: [0, 20])?.map { $0.rounded() } == [1, 0])
    }

    @Test func sparseCGLSMatchesDenseQRForCategoricalTerms() throws {
        let levels = Array(0..<8)
        var x: [[Double]] = []
        var y: [Double] = []
        for first in levels {
            for second in levels {
                for third in levels {
                    x.append([Double(first), Double(second), Double(third)])
                    y.append(1 + 0.25 * Double(first) - 0.1 * Double(second) + 0.05 * Double(third))
                }
            }
        }
        let terms: [MultivariateTermSpecification] = (0..<3).map {
            .categorical(.init(predictorIndex: $0, levels: levels, referenceLevel: 0))
        }
        let dense = try #require(MultivariateModel.fit(
            trainX: x, trainY: y, family: .gaussian,
            specification: .init(terms: terms, penaltyWeight: 0.1, solverPreference: .denseQR,
                                 maxIterations: 20, tolerance: 1e-9)
        ).model)

        #if canImport(NumericCoreSparse)
        let sparse = try #require(MultivariateModel.fit(
            trainX: x, trainY: y, family: .gaussian,
            specification: .init(terms: terms, penaltyWeight: 0.1, solverPreference: .sparseCGLS,
                                 maxIterations: 20, tolerance: 1e-9)
        ).model)
        #expect(sparse.solverBackend == .sparseCGLS)
        for point in [[0.0, 0, 0], [4, 2, 7], [7, 6, 1]] {
            #expect(abs(sparse.predict(point) - dense.predict(point)) < 1e-6)
        }
        #else
        #expect(dense.solverBackend == .denseQR)
        #endif
    }

    @Test func largeLowDensityFactorsAutomaticallySelectSparseCGLS() throws {
        #if canImport(NumericCoreSparse)
        let levels = Array(0..<64)
        let rowCount = 1_000
        var x: [[Double]] = []
        var y: [Double] = []
        for row in 0..<rowCount {
            let first = Double(row % levels.count)
            let second = Double((row / 2 + 17) % levels.count)
            let third = Double((row / 3 + 34) % levels.count)
            let fourth = Double((row / 4 + 51) % levels.count)
            x.append([first, second, third, fourth])
            y.append(1 + 0.2 * first / 63 - 0.15 * second / 63 + 0.1 * third / 63 - 0.05 * fourth / 63)
        }
        let terms: [MultivariateTermSpecification] = (0..<4).map {
            .categorical(.init(predictorIndex: $0, levels: levels, referenceLevel: 0))
        }
        let fit = try #require(MultivariateModel.fit(
            trainX: x, trainY: y, family: .gaussian,
            specification: .init(terms: terms, penaltyWeight: 0.1, maxIterations: 40, tolerance: 1e-8)
        ).model)
        #expect(fit.solverBackend == MultivariateSolverBackend.sparseCGLS)
        #expect(fit.deviance.isFinite && fit.deviance < fit.nullDeviance)
        let execution = try #require(fit.sparseExecution)
        #expect(execution.designRows == rowCount)
        #expect(execution.designColumns == 253)
        #expect(execution.nonZeroCount < execution.designRows * execution.designColumns / 8)
        #expect(execution.converged)
        #expect(execution.iterations > 0)
        #expect(execution.normalResidualNorm.isFinite)
        #expect(execution.workingObjective.isFinite)

        let unified = FittedStatisticalModel(
            multivariate: fit,
            specification: .init(strategy: .multivariateGaussian, multivariate: .init(
                terms: terms, penaltyWeight: 0.1, solverPreference: .automatic,
                maxIterations: 40, tolerance: 1e-8
            ))
        )
        #expect(unified.sparseExecution == execution)
        #endif
    }

    @Test func highCardinalitySparseFactorsUseBoundedDiagonalInference() throws {
        #if canImport(NumericCoreSparse)
        let levels = Array(0..<130)
        var x: [[Double]] = []
        var y: [Double] = []
        for row in 0..<1_300 {
            let first = row % levels.count
            let second = (row * 17 + row / levels.count) % levels.count
            x.append([Double(first), Double(second)])
            y.append(1 + 0.3 * Double(first) / 129 - 0.2 * Double(second) / 129)
        }
        let terms: [MultivariateTermSpecification] = [
            .categorical(.init(predictorIndex: 0, levels: levels, referenceLevel: 0)),
            .categorical(.init(predictorIndex: 1, levels: levels, referenceLevel: 0)),
        ]
        let fit = try #require(MultivariateModel.fit(
            trainX: x, trainY: y, family: .gaussian,
            specification: .init(
                terms: terms, penaltyWeight: 0.1, solverPreference: .sparseCGLS,
                maxIterations: 40, tolerance: 1e-8,
                sparseMaximumIterations: 8_000, sparseTolerance: 1e-9
            )
        ).model)
        #expect(fit.solverBackend == .sparseCGLS)
        #expect(fit.inference.method == .diagonalConditionalApproximation)
        #expect(fit.inference.coefficientCovariance.isEmpty)
        #expect(fit.inference.coefficientStandardErrors.count == 259)
        #expect(fit.standardError(at: [12, 25])?.isFinite == true)
        let execution = try #require(fit.sparseExecution)
        #expect(execution.designColumns == 259)
        #expect(execution.inferenceMethod == .diagonalConditionalApproximation)
        #expect(execution.workingSolveIterationLimit == 8_000)
        #expect(execution.workingSolveTolerance == 1e-9)
        #endif
    }

    @Test func phase13SpecificationsDecodeWithAutomaticSolverSelection() throws {
        let phase13JSON = """
        {"defaultKnotCount":3,"penaltyWeight":1,"maxIterations":50,"tolerance":1e-08}
        """
        let specification = try JSONDecoder().decode(
            MultivariateModelSpecification.self, from: Data(phase13JSON.utf8)
        )
        #expect(specification.terms == nil)
        #expect(specification.solverPreference == .automatic)
        #expect(specification.sparseMaximumIterations == 10_000)
        #expect(specification.sparseTolerance == 1e-10)
    }

    @Test func binomialTensorUsesIRLSAndUnifiedValidation() throws {
        var rng = SeedableRandomNumberGenerator(seed: 0x7135_2026)
        var x: [[Double]] = []
        var y: [Double] = []
        for first in 0..<14 {
            for second in 0..<12 {
                let a = -1 + Double(first) / 6.5
                let b = -1 + Double(second) / 5.5
                x.append([a, b])
                let probability = 1 / (1 + exp(-(-0.25 + 1.6 * a * b)))
                y.append(rng.nextBool(probability: probability) ? 1 : 0)
            }
        }
        let specification = MultivariateModelSpecification(
            terms: tensorTerms, penaltyWeight: 0.2, maxIterations: 100, tolerance: 1e-8
        )
        let fit = try #require(MultivariateModel.fit(
            trainX: x, trainY: y, family: .binomial, specification: specification
        ).model)
        #expect(fit.deviance < fit.nullDeviance)
        #expect(fit.predict([0.8, 0.8]) > fit.predict([0.8, -0.8]))
        #expect(fit.standardError(at: [0, 0]) != nil)

        let unifiedSpecification = StatisticalModelSpecification(
            strategy: .multivariateBinomial, multivariate: specification
        )
        let validation = try #require(CrossValidation.evaluate(
            trainX: x, trainY: y,
            configuration: .init(foldCount: 4, partitioning: .stratifiedBinary, seed: 5,
                                 specification: unifiedSpecification)
        ))
        #expect(validation.responseFamily == .binomial)
        #expect(validation.meanDeviance?.isFinite == true)
    }

    @Test func poissonTensorUsesTheDeclaredCountFamily() throws {
        var rng = SeedableRandomNumberGenerator(seed: 0xA015_2026)
        var x: [[Double]] = []
        var y: [Double] = []
        for first in 0..<12 {
            for second in 0..<11 {
                let a = -1 + Double(first) / 5.5
                let b = -1 + Double(second) / 5
                x.append([a, b])
                y.append(Double(poisson(mean: exp(0.3 + 0.75 * a * b), using: &rng)))
            }
        }
        let fit = try #require(MultivariateModel.fit(
            trainX: x, trainY: y, family: .poisson,
            specification: .init(terms: tensorTerms, penaltyWeight: 0.2,
                                 maxIterations: 100, tolerance: 1e-8)
        ).model)
        #expect(fit.deviance < fit.nullDeviance)
        #expect(fit.predict([0.8, 0.8]) > fit.predict([0.8, -0.8]))
        #expect(fit.diagnostics.responseFamily == .poisson)
    }

    @Test func spatialTemporalWorkflowIsSerializableAndUsesBlockedValidation() throws {
        var x: [[Double]] = []
        var y: [Double] = []
        for time in 0..<5 {
            for first in 0..<6 {
                for second in 0..<5 {
                    let a = -1 + Double(first) / 2.5
                    let b = -1 + Double(second) / 2
                    let t = Double(time) / 4
                    x.append([a, b, t])
                    y.append(1.5 + 0.8 * a * b + 0.6 * t)
                }
            }
        }
        let workflow = SpatialTemporalWorkflowSpecification(
            spatial: .init(firstPredictorIndex: 0, secondPredictorIndex: 1,
                           firstKnotCount: 0, secondKnotCount: 0),
            temporal: .init(predictorIndex: 2, knotCount: 0)
        )
        let multivariate = MultivariateModelSpecification(
            terms: [], spatialTemporal: workflow, defaultKnotCount: 0,
            penaltyWeight: 1e-6, maxIterations: 20, tolerance: 1e-9
        )
        let encoded = try JSONEncoder().encode(multivariate)
        #expect(try JSONDecoder().decode(MultivariateModelSpecification.self, from: encoded) == multivariate)
        let specification = StatisticalModelSpecification(
            strategy: .multivariateGaussian, multivariate: multivariate
        )
        let model = try #require(FittedStatisticalModel.fit(trainX: x, trainY: y, specification: specification))
        #expect(model.kind == .multivariateGaussian)
        #expect(abs(model.predict([0.25, -0.5, 0.6]) - (1.5 + 0.8 * 0.25 * -0.5 + 0.6 * 0.6)) < 1e-5)
        #expect(model.contour(xPredictorIndex: 0, yPredictorIndex: 1, baseline: [0, 0, 0.5]) != nil)
        let validation = try #require(CrossValidation.evaluate(
            trainX: x, trainY: y,
            configuration: .init(foldCount: 5, partitioning: .blocked, seed: 1,
                                 specification: specification)
        ))
        #expect(validation.responseFamily == .gaussian)
        #expect(validation.rootMeanSquaredError?.isFinite == true)
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
