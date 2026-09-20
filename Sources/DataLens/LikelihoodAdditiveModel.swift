import Foundation
#if canImport(NumericCoreAccelerate)
import NumericCoreAccelerate
#endif

/// Response family supported by a penalized likelihood additive model.
public enum LikelihoodAdditiveFamily: String, Codable, Sendable, Hashable {
    /// Bernoulli observations with a logit link. Responses must be exactly 0 or 1.
    case binomial
    /// Count observations with a log link. Responses must be finite and non-negative.
    case poisson
}

/// One continuous main-effect term in a likelihood additive model.
///
/// The term uses a centered cubic truncated-power regression-spline basis.
/// `knotCount` controls the number of evenly spaced interior knots; it is a
/// basis-complexity setting, while the model-wide penalty controls shrinkage.
public struct LikelihoodAdditiveTermSpecification: Codable, Sendable, Hashable {
    public let predictorIndex: Int
    public let knotCount: Int

    public init(predictorIndex: Int, knotCount: Int = 3) {
        self.predictorIndex = predictorIndex
        self.knotCount = knotCount
    }
}

/// Serializable configuration for a binomial or Poisson penalized GAM.
public struct LikelihoodAdditiveModelSpecification: Codable, Sendable, Hashable {
    public let terms: [LikelihoodAdditiveTermSpecification]?
    public let defaultKnotCount: Int
    /// Ridge penalty on every non-intercept regression-spline coefficient.
    public let penaltyWeight: Double
    public let maxIterations: Int
    public let tolerance: Double

    public init(
        terms: [LikelihoodAdditiveTermSpecification]? = nil,
        defaultKnotCount: Int = 3,
        penaltyWeight: Double = 1,
        maxIterations: Int = 50,
        tolerance: Double = 1e-8
    ) {
        self.terms = terms
        self.defaultKnotCount = defaultKnotCount
        self.penaltyWeight = penaltyWeight
        self.maxIterations = maxIterations
        self.tolerance = tolerance
    }

    var isValid: Bool {
        guard (0...12).contains(defaultKnotCount), penaltyWeight.isFinite,
              penaltyWeight > 0, maxIterations > 0, tolerance.isFinite, tolerance > 0
        else { return false }
        guard let terms else { return true }
        return !terms.isEmpty && terms.allSatisfy {
            $0.predictorIndex >= 0 && (0...12).contains($0.knotCount)
        }
    }
}

/// Why a likelihood-GAM fitting attempt stopped.
///
/// Only `.converged` exposes a model. All other outcomes intentionally keep a
/// partial IRLS iterate private, preventing a numerical approximation from
/// being mistaken for a fitted statistical model.
public enum LikelihoodAdditiveFitStatus: String, Codable, Sendable, Hashable {
    case converged
    case invalidInput
    case numericalFailure
    case lineSearchFailure
    case iterationLimit
}

/// Explicit outcome of a penalized IRLS likelihood-GAM fit.
public struct LikelihoodAdditiveFitResult: Sendable {
    public let status: LikelihoodAdditiveFitStatus
    public let iterations: Int
    public let deviance: Double?
    public let penalizedObjective: Double?
    public let scoreInfinityNorm: Double?
    /// Non-nil only when `status == .converged`.
    public let model: LikelihoodAdditiveModel?

    public var converged: Bool { status == .converged && model != nil }

    fileprivate init(
        status: LikelihoodAdditiveFitStatus, iterations: Int, deviance: Double? = nil,
        penalizedObjective: Double? = nil, scoreInfinityNorm: Double? = nil,
        model: LikelihoodAdditiveModel? = nil
    ) {
        self.status = status
        self.iterations = iterations
        self.deviance = deviance
        self.penalizedObjective = penalizedObjective
        self.scoreInfinityNorm = scoreInfinityNorm
        self.model = model
    }
}

/// A centered cubic regression-spline component in a likelihood GAM.
public struct LikelihoodAdditiveTerm: Sendable, Hashable {
    public let specification: LikelihoodAdditiveTermSpecification
    public let minimum: Double
    public let maximum: Double
    public let interiorKnots: [Double]
    public let coefficients: [Double]
    private let columnMeans: [Double]

    fileprivate init(
        specification: LikelihoodAdditiveTermSpecification, minimum: Double, maximum: Double,
        interiorKnots: [Double], columnMeans: [Double], coefficients: [Double]
    ) {
        self.specification = specification
        self.minimum = minimum
        self.maximum = maximum
        self.interiorKnots = interiorKnots
        self.columnMeans = columnMeans
        self.coefficients = coefficients
    }

    /// Centered contribution to the model's linear predictor at `value`.
    public func linearPredictorContribution(_ value: Double) -> Double {
        guard value.isFinite else { return .nan }
        return zip(centeredBasis(value), coefficients).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// Derivative of this term's linear-predictor contribution.
    public func linearPredictorGradient(_ value: Double) -> Double? {
        guard value.isFinite else { return nil }
        let range = maximum - minimum
        guard range.isFinite && range > 0 else { return nil }
        let u = (value - minimum) / range
        var derivative = [1 / range, 2 * u / range, 3 * u * u / range]
        derivative += interiorKnots.map { knot in
            let delta = max(u - knot, 0)
            return 3 * delta * delta / range
        }
        return zip(derivative, coefficients).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// Centered component curve on the linear-predictor scale.
    public func partialEffect(count: Int = 100) -> AdditivePartialEffect? {
        guard count >= 2, maximum > minimum else { return nil }
        let values = (0..<count).map { minimum + (maximum - minimum) * Double($0) / Double(count - 1) }
        return AdditivePartialEffect(
            predictorIndex: specification.predictorIndex,
            x: values,
            effect: values.map(linearPredictorContribution),
            gradient: values.map { linearPredictorGradient($0) ?? .nan }
        )
    }

    fileprivate func centeredBasis(_ value: Double) -> [Double] {
        zip(rawBasis(value), columnMeans).map { $0 - $1 }
    }

    fileprivate func rawBasis(_ value: Double) -> [Double] {
        let u = (value - minimum) / (maximum - minimum)
        return [u, u * u, u * u * u] + interiorKnots.map { pow(max(u - $0, 0), 3) }
    }
}

/// A converged binomial or Poisson generalized additive model.
///
/// The linear predictor is an intercept plus centered cubic regression-spline
/// main effects. Each IRLS update solves a penalized weighted least-squares
/// problem. A step-halving objective check and an unpenalized-score check are
/// both required before a model is exposed; use ``fit(trainX:trainY:family:specification:droppingMissing:)``
/// when the stop reason itself is important.
public struct LikelihoodAdditiveModel: Sendable {
    public let family: LikelihoodAdditiveFamily
    public let trainX: [[Double]]
    public let trainY: [Double]
    public let intercept: Double
    public let terms: [LikelihoodAdditiveTerm]
    public let fittedValues: [Double]
    public let linearPredictors: [Double]
    public let deviance: Double
    public let nullDeviance: Double
    public let penalizedObjective: Double
    public let penaltyWeight: Double
    public let iterations: Int
    public let maximumCoefficientChange: Double
    public let scoreInfinityNorm: Double
    public let keptIndices: [Int]

    private init(
        family: LikelihoodAdditiveFamily, trainX: [[Double]], trainY: [Double],
        intercept: Double, terms: [LikelihoodAdditiveTerm], fittedValues: [Double],
        linearPredictors: [Double], deviance: Double, nullDeviance: Double,
        penalizedObjective: Double, penaltyWeight: Double, iterations: Int,
        maximumCoefficientChange: Double, scoreInfinityNorm: Double, keptIndices: [Int]
    ) {
        self.family = family
        self.trainX = trainX
        self.trainY = trainY
        self.intercept = intercept
        self.terms = terms
        self.fittedValues = fittedValues
        self.linearPredictors = linearPredictors
        self.deviance = deviance
        self.nullDeviance = nullDeviance
        self.penalizedObjective = penalizedObjective
        self.penaltyWeight = penaltyWeight
        self.iterations = iterations
        self.maximumCoefficientChange = maximumCoefficientChange
        self.scoreInfinityNorm = scoreInfinityNorm
        self.keptIndices = keptIndices
    }

    /// Fit a likelihood GAM and report its explicit convergence outcome.
    public static func fit(
        trainX: [[Double]], trainY: [Double], family: LikelihoodAdditiveFamily,
        specification: LikelihoodAdditiveModelSpecification = LikelihoodAdditiveModelSpecification(),
        droppingMissing: Bool = false
    ) -> LikelihoodAdditiveFitResult {
        guard specification.isValid, trainX.count == trainY.count else {
            return .init(status: .invalidInput, iterations: 0)
        }
        let cleaned: ([[Double]], [Double], [Int]) = droppingMissing
            ? MissingData.dropping(trainX: trainX, trainY: trainY)
            : (trainX, trainY, Array(trainX.indices))
        let x = cleaned.0
        let y = cleaned.1
        guard !x.isEmpty, !x[0].isEmpty, x.allSatisfy({ $0.count == x[0].count }),
              x.flatMap({ $0 }).allSatisfy(\.isFinite), isValidResponse(y, family: family)
        else { return .init(status: .invalidInput, iterations: 0) }

        let specifications = specification.terms ?? x[0].indices.map {
            LikelihoodAdditiveTermSpecification(
                predictorIndex: $0, knotCount: specification.defaultKnotCount
            )
        }
        guard !specifications.isEmpty,
              Set(specifications.map(\.predictorIndex)).count == specifications.count,
              specifications.allSatisfy({ x[0].indices.contains($0.predictorIndex) })
        else { return .init(status: .invalidInput, iterations: 0) }

        guard let basis = makeBasis(rows: x, specifications: specifications) else {
            return .init(status: .invalidInput, iterations: 0)
        }
        let design = basis.design
        let penalty = penaltyRows(columnCount: design[0].count)
        var coefficients = [Double](repeating: 0, count: design[0].count)
        coefficients[0] = startIntercept(y, family: family)
        var current = objective(design: design, response: y, coefficients: coefficients,
                                family: family, penaltyWeight: specification.penaltyWeight)
        guard current.isFinite else { return .init(status: .numericalFailure, iterations: 0) }
        let initialScore = scoreInfinityNorm(design: design, response: y, coefficients: coefficients,
                                             family: family, penaltyWeight: specification.penaltyWeight)
        guard initialScore.isFinite else { return .init(status: .numericalFailure, iterations: 0) }
        let scoreScale = max(1, initialScore)

        for iteration in 1...specification.maxIterations {
            let eta = design.map { dot($0, coefficients) }
            let working = eta.indices.map { workingPoint(eta: eta[$0], family: family) }
            let weights = working.map(\.weight)
            let response = eta.indices.map { eta[$0] + (y[$0] - working[$0].mean) / working[$0].derivative }
            guard weights.allSatisfy({ $0.isFinite && $0 > 0 }), response.allSatisfy(\.isFinite),
                  let proposal = solvePenalizedWorkingLeastSquares(
                    design: design, response: response, weights: weights, penalty: penalty,
                    // StatisticalSolver minimizes Σw(z-Xβ)² + λ‖Pβ‖².
                    // The target here is ½·deviance + λ‖Pβ‖², whose IRLS
                    // quadratic therefore needs twice the penalty weight.
                    penaltyWeight: 2 * specification.penaltyWeight
                  )
            else {
                return .init(status: .numericalFailure, iterations: iteration - 1,
                             deviance: current.deviance, penalizedObjective: current.value)
            }

            var step = 1.0
            var accepted: (coefficients: [Double], objective: Objective)?
            for _ in 0..<20 {
                let candidate = zip(coefficients, proposal).map { $0 + step * ($1 - $0) }
                let candidateObjective = objective(
                    design: design, response: y, coefficients: candidate, family: family,
                    penaltyWeight: specification.penaltyWeight
                )
                if candidateObjective.isFinite,
                   candidateObjective.value <= current.value + 1e-12 * max(1, current.value) {
                    accepted = (candidate, candidateObjective)
                    break
                }
                step *= 0.5
            }
            guard let accepted else {
                return .init(status: .lineSearchFailure, iterations: iteration - 1,
                             deviance: current.deviance, penalizedObjective: current.value)
            }

            let maximumChange = zip(coefficients, accepted.coefficients).map {
                abs($1 - $0)
            }.max() ?? 0
            coefficients = accepted.coefficients
            current = accepted.objective
            let score = scoreInfinityNorm(design: design, response: y, coefficients: coefficients,
                                          family: family, penaltyWeight: specification.penaltyWeight)
            guard score.isFinite else {
                return .init(status: .numericalFailure, iterations: iteration,
                             deviance: current.deviance, penalizedObjective: current.value)
            }
            let coefficientScale = max(1, coefficients.map(abs).max() ?? 0)
            let scoreTolerance = max(specification.tolerance, 1e-7) * scoreScale
            if maximumChange <= specification.tolerance * coefficientScale && score <= scoreTolerance {
                let linearPredictors = design.map { dot($0, coefficients) }
                let means = linearPredictors.map { mean(eta: $0, family: family) }
                let terms = basis.terms.enumerated().map { offset, component in
                    let range = basis.ranges[offset]
                    return LikelihoodAdditiveTerm(
                        specification: component.specification, minimum: component.minimum,
                        maximum: component.maximum, interiorKnots: component.interiorKnots,
                        columnMeans: component.columnMeans,
                        coefficients: Array(coefficients[range])
                    )
                }
                let model = LikelihoodAdditiveModel(
                    family: family, trainX: x, trainY: y, intercept: coefficients[0], terms: terms,
                    fittedValues: means, linearPredictors: linearPredictors,
                    deviance: current.deviance, nullDeviance: nullDeviance(y, family: family),
                    penalizedObjective: current.value, penaltyWeight: specification.penaltyWeight,
                    iterations: iteration, maximumCoefficientChange: maximumChange,
                    scoreInfinityNorm: score, keptIndices: cleaned.2
                )
                return .init(status: .converged, iterations: iteration, deviance: current.deviance,
                             penalizedObjective: current.value, scoreInfinityNorm: score, model: model)
            }
        }
        return .init(status: .iterationLimit, iterations: specification.maxIterations,
                     deviance: current.deviance, penalizedObjective: current.value)
    }

    /// Convenience form that returns a model only when IRLS converged.
    public static func fitConverged(
        trainX: [[Double]], trainY: [Double], family: LikelihoodAdditiveFamily,
        specification: LikelihoodAdditiveModelSpecification = LikelihoodAdditiveModelSpecification(),
        droppingMissing: Bool = false
    ) -> LikelihoodAdditiveModel? {
        fit(trainX: trainX, trainY: trainY, family: family, specification: specification,
            droppingMissing: droppingMissing).model
    }

    /// Fitted mean at a complete predictor row.
    public func predict(_ x: [Double]) -> Double {
        guard let eta = linearPredictor(x) else { return .nan }
        return Self.mean(eta: eta, family: family)
    }

    /// Fitted means at several complete predictor rows.
    public func predict(_ xs: [[Double]]) -> [Double] { xs.map(predict) }

    /// Additive component contributions on the linear-predictor scale.
    public func componentContributions(at x: [Double]) -> [Double]? {
        guard x.count == trainX[0].count, x.allSatisfy(\.isFinite) else { return nil }
        return terms.map { $0.linearPredictorContribution(x[$0.specification.predictorIndex]) }
    }

    /// Linear predictor at a complete row.
    public func linearPredictor(_ x: [Double]) -> Double? {
        componentContributions(at: x).map { intercept + $0.reduce(0, +) }
    }

    /// Gradient of the fitted mean with respect to each original predictor.
    public func gradient(at x: [Double]) -> [Double]? {
        guard let eta = linearPredictor(x), x.count == trainX[0].count else { return nil }
        let multiplier = Self.workingPoint(eta: eta, family: family).derivative
        var result = [Double](repeating: 0, count: x.count)
        for term in terms {
            let index = term.specification.predictorIndex
            guard let derivative = term.linearPredictorGradient(x[index]) else { return nil }
            result[index] = multiplier * derivative
        }
        return result
    }

    /// A centered component curve on the linear-predictor scale.
    public func partialEffect(forPredictor predictorIndex: Int, count: Int = 100) -> AdditivePartialEffect? {
        terms.first(where: { $0.specification.predictorIndex == predictorIndex })?.partialEffect(count: count)
    }

    /// Family-correct residuals aligned with `keptIndices`.
    public func residuals(_ kind: ResidualKind) -> [Double] {
        zip(trainY, fittedValues).map { y, mu in
            let raw = y - mu
            switch kind {
            case .raw: return raw
            case .pearson:
                switch family {
                case .binomial: return raw / sqrt(max(mu * (1 - mu), 1e-12))
                case .poisson: return raw / sqrt(max(mu, 1e-12))
                }
            case .deviance:
                let sign = raw < 0 ? -1.0 : raw > 0 ? 1.0 : 0
                return sign * sqrt(max(Self.unitDeviance(y: y, mean: mu, family: family), 0))
            }
        }
    }

    /// Diagnostics suitable for the unified-model contract.
    public var diagnostics: FitDiagnostics {
        FitDiagnostics(
            responseFamily: family == .binomial ? .binomial : .poisson,
            linkFunction: family == .binomial ? .logit : .log,
            observationCount: trainY.count,
            effectiveDegreesOfFreedom: Double(1 + terms.reduce(0) { $0 + $1.coefficients.count }),
            residualScale: 1, deviance: deviance, nullDeviance: nullDeviance
        )
    }

    private struct WorkingPoint {
        let mean: Double
        let derivative: Double
        let weight: Double
    }

    private struct Objective {
        let value: Double
        let deviance: Double
        var isFinite: Bool { value.isFinite && deviance.isFinite }
    }

    private struct BasisTerm {
        let specification: LikelihoodAdditiveTermSpecification
        let minimum: Double
        let maximum: Double
        let interiorKnots: [Double]
        let columnMeans: [Double]
    }

    private static func makeBasis(
        rows: [[Double]], specifications: [LikelihoodAdditiveTermSpecification]
    ) -> (design: [[Double]], terms: [BasisTerm], ranges: [Range<Int>])? {
        var terms: [BasisTerm] = []
        var ranges: [Range<Int>] = []
        var columns: [[Double]] = [[Double](repeating: 1, count: rows.count)]
        for specification in specifications {
            let values = rows.map { $0[specification.predictorIndex] }
            guard let minimum = values.min(), let maximum = values.max(), maximum > minimum else { return nil }
            let knots = (1...specification.knotCount).map {
                Double($0) / Double(specification.knotCount + 1)
            }
            let raw = values.map { value in
                let u = (value - minimum) / (maximum - minimum)
                return [u, u * u, u * u * u] + knots.map { pow(max(u - $0, 0), 3) }
            }
            let means = raw[0].indices.map { index in raw.map { $0[index] }.reduce(0, +) / Double(rows.count) }
            let start = columns.count
            for index in means.indices { columns.append(raw.map { $0[index] - means[index] }) }
            ranges.append(start..<columns.count)
            terms.append(BasisTerm(specification: specification, minimum: minimum, maximum: maximum,
                                   interiorKnots: knots, columnMeans: means))
        }
        let design = rows.indices.map { row in columns.map { $0[row] } }
        return (design, terms, ranges)
    }

    private static func penaltyRows(columnCount: Int) -> [[Double]] {
        (1..<columnCount).map { index in
            var row = [Double](repeating: 0, count: columnCount)
            row[index] = 1
            return row
        }
    }

    private static func solvePenalizedWorkingLeastSquares(
        design: [[Double]], response: [Double], weights: [Double], penalty: [[Double]], penaltyWeight: Double
    ) -> [Double]? {
        #if canImport(NumericCoreAccelerate)
        return StatisticalSolver.penalizedWeightedLeastSquares(
            design: design, response: response, weights: weights,
            penaltyRows: penalty, penaltyWeight: penaltyWeight
        )?.coefficients
        #else
        var augmented = [[Double]]()
        var augmentedResponse = [Double]()
        for index in design.indices {
            let scale = sqrt(weights[index])
            augmented.append(design[index].map { scale * $0 })
            augmentedResponse.append(scale * response[index])
        }
        let penaltyScale = sqrt(penaltyWeight)
        for row in penalty {
            augmented.append(row.map { penaltyScale * $0 })
            augmentedResponse.append(0)
        }
        return LinAlg.leastSquares(augmented, augmentedResponse)
        #endif
    }

    private static func isValidResponse(_ values: [Double], family: LikelihoodAdditiveFamily) -> Bool {
        switch family {
        case .binomial: values.allSatisfy { $0 == 0 || $0 == 1 }
        case .poisson: values.allSatisfy { $0.isFinite && $0 >= 0 }
        }
    }

    private static func mean(eta: Double, family: LikelihoodAdditiveFamily) -> Double {
        let clamped = min(max(eta, -30), 30)
        switch family {
        case .binomial:
            return clamped >= 0 ? 1 / (1 + exp(-clamped)) : exp(clamped) / (1 + exp(clamped))
        case .poisson:
            return max(exp(clamped), 1e-12)
        }
    }

    private static func workingPoint(eta: Double, family: LikelihoodAdditiveFamily) -> WorkingPoint {
        let mean = mean(eta: eta, family: family)
        switch family {
        case .binomial:
            let derivative = max(mean * (1 - mean), 1e-12)
            return WorkingPoint(mean: mean, derivative: derivative, weight: derivative)
        case .poisson:
            return WorkingPoint(mean: mean, derivative: mean, weight: mean)
        }
    }

    private static func startIntercept(_ values: [Double], family: LikelihoodAdditiveFamily) -> Double {
        let mean = values.reduce(0, +) / Double(values.count)
        switch family {
        case .binomial:
            let probability = min(max(mean, 1e-6), 1 - 1e-6)
            return log(probability / (1 - probability))
        case .poisson:
            return log(max(mean, 1e-6))
        }
    }

    private static func objective(
        design: [[Double]], response: [Double], coefficients: [Double],
        family: LikelihoodAdditiveFamily, penaltyWeight: Double
    ) -> Objective {
        let deviance = zip(design, response).reduce(0.0) {
            $0 + unitDeviance(y: $1.1, mean: mean(eta: dot($1.0, coefficients), family: family), family: family)
        }
        let penalty = coefficients.dropFirst().reduce(0.0) { $0 + $1 * $1 }
        return Objective(value: 0.5 * deviance + penaltyWeight * penalty, deviance: deviance)
    }

    private static func nullDeviance(_ response: [Double], family: LikelihoodAdditiveFamily) -> Double {
        let eta = startIntercept(response, family: family)
        let mean = mean(eta: eta, family: family)
        return response.reduce(0.0) { $0 + unitDeviance(y: $1, mean: mean, family: family) }
    }

    private static func unitDeviance(y: Double, mean: Double, family: LikelihoodAdditiveFamily) -> Double {
        switch family {
        case .binomial:
            let m = min(max(mean, 1e-12), 1 - 1e-12)
            let first = y == 0 ? 0 : y * log(y / m)
            let second = y == 1 ? 0 : (1 - y) * log((1 - y) / (1 - m))
            return 2 * (first + second)
        case .poisson:
            let m = max(mean, 1e-12)
            return y == 0 ? 2 * m : 2 * (y * log(y / m) - (y - m))
        }
    }

    private static func scoreInfinityNorm(
        design: [[Double]], response: [Double], coefficients: [Double],
        family: LikelihoodAdditiveFamily, penaltyWeight: Double
    ) -> Double {
        var score = [Double](repeating: 0, count: coefficients.count)
        for (row, y) in zip(design, response) {
            let residual = y - mean(eta: dot(row, coefficients), family: family)
            for index in score.indices { score[index] += row[index] * residual }
        }
        for index in score.indices.dropFirst() { score[index] -= 2 * penaltyWeight * coefficients[index] }
        return score.map(abs).max() ?? .infinity
    }

    private static func dot(_ left: [Double], _ right: [Double]) -> Double {
        zip(left, right).reduce(0) { $0 + $1.0 * $1.1 }
    }
}
