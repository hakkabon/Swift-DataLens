import Foundation
import Testing
@testable import DataLens

@Suite("Additive model")
struct AdditiveModelTests {
    private func fixture() -> (x: [[Double]], y: [Double], truth: [Double]) {
        var x: [[Double]] = []
        var y: [Double] = []
        var truth: [Double] = []
        for i in 0..<13 {
            let a = -2.0 + 4.0 * Double(i) / 12
            for j in 0..<11 {
                let b = -1.5 + 3.0 * Double(j) / 10
                let mean = 2.5 + sin(a) + 0.45 * b * b
                x.append([a, b])
                y.append(mean)
                truth.append(mean)
            }
        }
        return (x, y, truth)
    }

    @Test func recoversAdditiveTruthAndCentersTerms() {
        let data = fixture()
        let fit = AdditiveModel.fit(trainX: data.x, trainY: data.y,
                                    defaultSpan: 0.45, defaultDegree: 2)!
        let rmse = sqrt(zip(fit.fittedValues, data.truth).reduce(0.0) {
            $0 + pow($1.0 - $1.1, 2)
        } / Double(data.truth.count))
        #expect(rmse < 0.04)
        #expect(fit.iterations <= 10)
        #expect(fit.maximumChange <= 1e-8 * max(1, data.y.map { abs($0 - fit.intercept) }.max()!))
        #expect(fit.terms.count == 2)
        for term in fit.terms {
            let values = data.x.map { term.predict($0[term.specification.predictorIndex]) }
            #expect(abs(values.reduce(0, +) / Double(values.count)) < 1e-10)
        }
    }

    @Test func decompositionPredictionAndGradientAgree() {
        let data = fixture()
        let fit = AdditiveModel.fit(trainX: data.x, trainY: data.y,
                                    defaultSpan: 0.5, defaultDegree: 2)!
        let point = [0.37, -0.41]
        let effects = fit.componentContributions(at: point)!
        #expect(abs(fit.predict(point) - fit.intercept - effects.reduce(0, +)) < 1e-12)
        #expect(fit.predict([point, point]) == [fit.predict(point), fit.predict(point)])

        let gradient = fit.gradient(at: point)!
        for term in fit.terms {
            let j = term.specification.predictorIndex
            #expect(gradient[j] == term.gradient(at: point[j]))
        }
        // Local-polynomial gradients estimate the derivatives of the two
        // generating effects (they are not finite differences of the
        // weight-varying prediction function).
        #expect(abs(gradient[0] - cos(point[0])) < 0.08)
        #expect(abs(gradient[1] - 0.9 * point[1]) < 0.08)
    }

    @Test func selectedTermsLeaveOtherGradientCoordinatesZero() {
        let data = fixture()
        let spec = AdditiveTermSpecification(predictorIndex: 1, span: 0.5, degree: 2)
        let fit = AdditiveModel.fit(trainX: data.x, trainY: data.y, terms: [spec])!
        let gradient = fit.gradient(at: [0.2, 0.3])!
        #expect(gradient[0] == 0)
        #expect(gradient[1].isFinite)
    }

    @Test func droppingMissingIsWholeRowAndRecordsIndices() {
        let data = fixture()
        var x = data.x
        var y = data.y
        x[3][1] = .nan
        y[17] = .nan
        let fit = AdditiveModel.fit(trainX: x, trainY: y, defaultSpan: 0.5,
                                    droppingMissing: true)!
        #expect(fit.keptIndices == data.x.indices.filter { $0 != 3 && $0 != 17 })
        #expect(fit.fittedValues.count == data.x.count - 2)
        #expect(fit.fittedValues.allSatisfy { $0.isFinite })
    }

    @Test func validationAndConvergenceFailure() {
        let data = fixture()
        #expect(AdditiveModel.fit(trainX: [], trainY: []) == nil)
        #expect(AdditiveModel.fit(trainX: [[0, 1]], trainY: [1, 2]) == nil)
        #expect(AdditiveModel.fit(trainX: [[0], [1, 2]], trainY: [0, 1]) == nil)
        #expect(AdditiveModel.fit(trainX: data.x, trainY: data.y, defaultSpan: 0) == nil)
        #expect(AdditiveModel.fit(trainX: data.x, trainY: data.y, tolerance: 0) == nil)
        #expect(AdditiveModel.fit(trainX: data.x, trainY: data.y,
                                  terms: [.init(predictorIndex: 2)]) == nil)
        #expect(AdditiveModel.fit(trainX: data.x, trainY: data.y,
                                  terms: [.init(predictorIndex: 0),
                                          .init(predictorIndex: 0)]) == nil)
        // A deliberately inadequate iteration cap must not leak a partial fit.
        #expect(AdditiveModel.fit(trainX: data.x, trainY: data.y,
                                  maxIterations: 1, tolerance: 1e-14) == nil)
        let fit = AdditiveModel.fit(trainX: data.x, trainY: data.y)!
        #expect(fit.predict([0]).isNaN)
        #expect(fit.componentContributions(at: [.nan, 0]) == nil)
    }
}
