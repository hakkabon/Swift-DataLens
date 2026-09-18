import Foundation
import Testing
@testable import DataLens

@Suite("Fit diagnostics")
struct FitDiagnosticsTests {
    @Test func gaussianMetadataAndResiduals() throws {
        let xs = (0..<12).map { [Double($0)] }
        let ys = (0..<12).map { 2 * Double($0) + (Double($0 % 2) - 0.5) }
        let fit = try #require(Loess.fit(trainX: xs, trainY: ys, span: 0.75, degree: 1,
                                         robustIterations: 0))
        let erased = FittedSmoother.loess(fit)

        #expect(erased.diagnostics.responseFamily == .gaussian)
        #expect(erased.diagnostics.linkFunction == .identity)
        #expect(erased.diagnostics.observationCount == ys.count)
        #expect(erased.residuals(.raw) == zip(ys, fit.fittedValues).map { $0 - $1 })
        #expect(erased.residuals(.deviance) == erased.residuals(.raw))
        #expect(erased.residuals(.pearson).allSatisfy { $0.isFinite })
    }

    @Test func binomialDevianceResidualsReconstructDeviance() throws {
        let xs = (0..<30).map { [Double($0) / 5] }
        let ys = (0..<30).map { $0 < 13 ? 0.0 : 1.0 }
        let fit = try #require(LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                                   family: .binomial, span: 0.6))
        let erased = FittedSmoother.likelihood(fit)
        let residuals = erased.residuals(.deviance)

        #expect(erased.diagnostics.responseFamily == .binomial)
        #expect(erased.diagnostics.linkFunction == .logit)
        #expect(abs(residuals.reduce(0) { $0 + $1 * $1 } - fit.deviance) < 1e-9)
        #expect(residuals.allSatisfy { $0.isFinite })
    }

    @Test func poissonDevianceResidualsReconstructDeviance() throws {
        let xs = (0..<24).map { [Double($0)] }
        let ys = (0..<24).map { Double(($0 * 3) % 7) }
        let fit = try #require(LocalLikelihood.fit(trainX: xs, trainY: ys, degree: 1,
                                                   family: .poisson, span: 0.75))
        let erased = FittedSmoother.likelihood(fit)
        let residuals = erased.residuals(.deviance)

        #expect(erased.diagnostics.responseFamily == .poisson)
        #expect(erased.diagnostics.linkFunction == .log)
        #expect(abs(residuals.reduce(0) { $0 + $1 * $1 } - fit.deviance) < 1e-8)
        #expect(erased.residuals(.pearson).allSatisfy { $0.isFinite })
    }

    @Test func metadataRoundTripsThroughJSON() throws {
        let value = FitDiagnostics(responseFamily: .poisson, linkFunction: .log,
                                   observationCount: 42, effectiveDegreesOfFreedom: 6.5,
                                   residualScale: 1, deviance: 12, nullDeviance: 30)
        let decoded = try JSONDecoder().decode(FitDiagnostics.self,
                                               from: JSONEncoder().encode(value))
        #expect(decoded == value)
    }
}
