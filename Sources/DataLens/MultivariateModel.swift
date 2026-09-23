import Foundation
#if canImport(NumericCoreAccelerate)
import NumericCoreAccelerate
#endif
#if canImport(NumericCoreSparse)
import NumericCoreSparse
#endif

/// Response family supported by ``MultivariateModel``.
public enum MultivariateResponseFamily: String, Codable, Sendable, Hashable {
    case gaussian
    case binomial
    case poisson

    fileprivate var responseFamily: ResponseFamily {
        switch self {
        case .gaussian: .gaussian
        case .binomial: .binomial
        case .poisson: .poisson
        }
    }
}

/// Requested numerical path for a multivariate penalized least-squares update.
public enum MultivariateSolverPreference: String, Codable, Sendable, Hashable {
    /// Use sparse CGLS only for profiled, sufficiently sparse large designs.
    case automatic
    /// Always use the rank-revealing dense augmented-QR path.
    case denseQR
    /// Require the portable CSR CGLS path; a non-converged sparse solve fails closed.
    case sparseCGLS
}

/// Numerical backend actually used by a converged multivariate fit.
public enum MultivariateSolverBackend: String, Codable, Sendable, Hashable {
    case denseQR
    /// Rust-NumericCore CSR CGLS through Swift-NumericCore's sparse bridge.
    case sparseCGLS
    /// Automatic sparse dispatch did not converge, so dense QR supplied the checked fit.
    case denseQRFallback
}

/// Conditional-inference route retained with a multivariate fit.
///
/// Full dense covariance remains the exact fixed-basis calculation for compact
/// designs. Large native sparse fits use a diagonal approximation instead of
/// allocating a dense covariance matrix whose size would defeat sparse
/// execution; intervals then remain explicitly conditional and approximate.
public enum SparseInferenceMethod: String, Codable, Sendable, Hashable {
    case exactDenseCovariance
    case diagonalConditionalApproximation
}

/// Auditable outcome of the final native CSR statistical solve.
///
/// This records the numerical work actually accepted by a multivariate fit,
/// rather than merely retaining a solver preference. It is present only when
/// ``MultivariateSolverBackend/sparseCGLS`` supplied the final update. The
/// residual is CGLS's relative normal-equation residual; it is not a model
/// residual or a goodness-of-fit statistic.
public struct SparseExecutionEvidence: Codable, Sendable, Hashable {
    /// Shape of the CSR design supplied to Rust-NumericCore.
    public let designRows: Int
    public let designColumns: Int
    public let nonZeroCount: Int
    public let iterations: Int
    /// Relative normal-equation residual reported by CGLS.
    public let normalResidualNorm: Double
    public let converged: Bool
    /// Weighted residual sum of squares for CGLS's final working solve.
    public let weightedResidualSumOfSquares: Double
    /// Penalty portion of CGLS's final working objective.
    public let penaltyContribution: Double
    /// Iteration budget supplied to the accepted CGLS working solve. `nil`
    /// means a legacy evidence record did not retain this setting.
    public let workingSolveIterationLimit: Int?
    /// Relative normal-residual tolerance supplied to the working solve.
    /// `nil` means a legacy record did not retain it.
    public let workingSolveTolerance: Double?
    /// Inference route retained with the fitted coefficients.
    public let inferenceMethod: SparseInferenceMethod

    /// `weightedResidualSumOfSquares + penaltyContribution` for the final
    /// working least-squares update. It is intentionally distinct from a
    /// model family's deviance and penalized likelihood.
    public var workingObjective: Double {
        weightedResidualSumOfSquares + penaltyContribution
    }

    fileprivate init(
        designRows: Int, designColumns: Int, nonZeroCount: Int,
        iterations: Int, normalResidualNorm: Double, converged: Bool,
        weightedResidualSumOfSquares: Double, penaltyContribution: Double,
        workingSolveIterationLimit: Int, workingSolveTolerance: Double,
        inferenceMethod: SparseInferenceMethod
    ) {
        self.designRows = designRows
        self.designColumns = designColumns
        self.nonZeroCount = nonZeroCount
        self.iterations = iterations
        self.normalResidualNorm = normalResidualNorm
        self.converged = converged
        self.weightedResidualSumOfSquares = weightedResidualSumOfSquares
        self.penaltyContribution = penaltyContribution
        self.workingSolveIterationLimit = workingSolveIterationLimit
        self.workingSolveTolerance = workingSolveTolerance
        self.inferenceMethod = inferenceMethod
    }

    private enum CodingKeys: String, CodingKey {
        case designRows, designColumns, nonZeroCount, iterations, normalResidualNorm, converged
        case weightedResidualSumOfSquares, penaltyContribution
        case workingSolveIterationLimit, workingSolveTolerance, inferenceMethod
    }

    /// Reads Phase 14 sparse records that predate explicit CGLS settings
    /// without fabricating an unknown tolerance.
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        designRows = try values.decode(Int.self, forKey: .designRows)
        designColumns = try values.decode(Int.self, forKey: .designColumns)
        nonZeroCount = try values.decode(Int.self, forKey: .nonZeroCount)
        iterations = try values.decode(Int.self, forKey: .iterations)
        normalResidualNorm = try values.decode(Double.self, forKey: .normalResidualNorm)
        converged = try values.decode(Bool.self, forKey: .converged)
        weightedResidualSumOfSquares = try values.decode(Double.self, forKey: .weightedResidualSumOfSquares)
        penaltyContribution = try values.decode(Double.self, forKey: .penaltyContribution)
        workingSolveIterationLimit = try values.decodeIfPresent(
            Int.self, forKey: .workingSolveIterationLimit
        )
        workingSolveTolerance = try values.decodeIfPresent(
            Double.self, forKey: .workingSolveTolerance
        )
        inferenceMethod = try values.decodeIfPresent(SparseInferenceMethod.self, forKey: .inferenceMethod)
            ?? .exactDenseCovariance
    }
}

/// One continuous univariate regression-spline main effect.
public struct SplineTermSpecification: Codable, Sendable, Hashable {
    public let predictorIndex: Int
    public let knotCount: Int

    public init(predictorIndex: Int, knotCount: Int = 3) {
        self.predictorIndex = predictorIndex
        self.knotCount = knotCount
    }
}

/// A tensor-product regression-spline interaction between two continuous predictors.
///
/// The interaction includes the full tensor basis. Include separate
/// ``SplineTermSpecification`` values when main effects are also desired.
public struct TensorProductTermSpecification: Codable, Sendable, Hashable {
    public let firstPredictorIndex: Int
    public let secondPredictorIndex: Int
    public let firstKnotCount: Int
    public let secondKnotCount: Int

    public init(
        firstPredictorIndex: Int, secondPredictorIndex: Int,
        firstKnotCount: Int = 2, secondKnotCount: Int = 2
    ) {
        self.firstPredictorIndex = firstPredictorIndex
        self.secondPredictorIndex = secondPredictorIndex
        self.firstKnotCount = firstKnotCount
        self.secondKnotCount = secondKnotCount
    }
}

/// A treatment-coded categorical main effect.
///
/// Predictor values must be finite integer codes. Omit `levels` to infer the
/// retained training levels; otherwise every retained code must appear in it.
/// The smallest inferred level, or explicit `referenceLevel`, is the reference.
public struct CategoricalTermSpecification: Codable, Sendable, Hashable {
    public let predictorIndex: Int
    public let levels: [Int]?
    public let referenceLevel: Int?

    public init(predictorIndex: Int, levels: [Int]? = nil, referenceLevel: Int? = nil) {
        self.predictorIndex = predictorIndex
        self.levels = levels
        self.referenceLevel = referenceLevel
    }
}

/// A term in a generalized multivariate spline model.
public enum MultivariateTermSpecification: Codable, Sendable, Hashable {
    case spline(SplineTermSpecification)
    case tensorProduct(TensorProductTermSpecification)
    case categorical(CategoricalTermSpecification)
}

/// Semantic description of a spatial tensor surface and optional temporal spline.
///
/// It resolves into spatial marginal splines plus their tensor interaction,
/// and an optional temporal spline, preserving one fitting and validation path. Use blocked folds in ``CrossValidation`` when time order
/// is a forecasting boundary rather than an exchangeable covariate.
public struct SpatialTemporalWorkflowSpecification: Codable, Sendable, Hashable {
    public let spatial: TensorProductTermSpecification
    public let temporal: SplineTermSpecification?

    public init(spatial: TensorProductTermSpecification, temporal: SplineTermSpecification? = nil) {
        self.spatial = spatial
        self.temporal = temporal
    }
}

/// Serializable configuration for a generalized multivariate spline model.
public struct MultivariateModelSpecification: Codable, Sendable, Hashable {
    /// Explicit terms. `nil` selects a spline main effect for every predictor.
    public let terms: [MultivariateTermSpecification]?
    /// Optional semantic spatial/temporal terms appended to `terms`.
    public let spatialTemporal: SpatialTemporalWorkflowSpecification?
    public let defaultKnotCount: Int
    /// Ridge penalty applied to every non-intercept coefficient.
    public let penaltyWeight: Double
    /// Numerical path selection; `.automatic` preserves dense QR except for profiled sparse designs.
    public let solverPreference: MultivariateSolverPreference
    public let maxIterations: Int
    public let tolerance: Double
    /// CGLS iteration budget for each sparse WLS/IRLS working solve.
    public let sparseMaximumIterations: Int
    /// CGLS relative normal-residual tolerance, independent of IRLS convergence.
    public let sparseTolerance: Double

    public init(
        terms: [MultivariateTermSpecification]? = nil,
        spatialTemporal: SpatialTemporalWorkflowSpecification? = nil,
        defaultKnotCount: Int = 3, penaltyWeight: Double = 1,
        solverPreference: MultivariateSolverPreference = .automatic,
        maxIterations: Int = 50, tolerance: Double = 1e-8,
        sparseMaximumIterations: Int = 10_000, sparseTolerance: Double = 1e-10
    ) {
        self.terms = terms
        self.spatialTemporal = spatialTemporal
        self.defaultKnotCount = defaultKnotCount
        self.penaltyWeight = penaltyWeight
        self.solverPreference = solverPreference
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.sparseMaximumIterations = sparseMaximumIterations
        self.sparseTolerance = sparseTolerance
    }

    private enum CodingKeys: String, CodingKey {
        case terms, spatialTemporal, defaultKnotCount, penaltyWeight
        case solverPreference, maxIterations, tolerance, sparseMaximumIterations, sparseTolerance
    }

    /// Decodes Phase 13 saved specifications with `.automatic` acceleration,
    /// preserving replayability after the Phase 14 field was introduced.
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        terms = try values.decodeIfPresent([MultivariateTermSpecification].self, forKey: .terms)
        spatialTemporal = try values.decodeIfPresent(SpatialTemporalWorkflowSpecification.self, forKey: .spatialTemporal)
        defaultKnotCount = try values.decode(Int.self, forKey: .defaultKnotCount)
        penaltyWeight = try values.decode(Double.self, forKey: .penaltyWeight)
        solverPreference = try values.decodeIfPresent(MultivariateSolverPreference.self, forKey: .solverPreference) ?? .automatic
        maxIterations = try values.decode(Int.self, forKey: .maxIterations)
        tolerance = try values.decode(Double.self, forKey: .tolerance)
        sparseMaximumIterations = try values.decodeIfPresent(Int.self, forKey: .sparseMaximumIterations)
            ?? 10_000
        sparseTolerance = try values.decodeIfPresent(Double.self, forKey: .sparseTolerance)
            ?? 1e-10
    }

    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encodeIfPresent(terms, forKey: .terms)
        try values.encodeIfPresent(spatialTemporal, forKey: .spatialTemporal)
        try values.encode(defaultKnotCount, forKey: .defaultKnotCount)
        try values.encode(penaltyWeight, forKey: .penaltyWeight)
        try values.encode(solverPreference, forKey: .solverPreference)
        try values.encode(maxIterations, forKey: .maxIterations)
        try values.encode(tolerance, forKey: .tolerance)
        try values.encode(sparseMaximumIterations, forKey: .sparseMaximumIterations)
        try values.encode(sparseTolerance, forKey: .sparseTolerance)
    }

    var isValid: Bool {
        guard (0...8).contains(defaultKnotCount), penaltyWeight.isFinite, penaltyWeight > 0,
              maxIterations > 0, tolerance.isFinite, tolerance > 0,
              sparseMaximumIterations > 0, sparseTolerance.isFinite, sparseTolerance > 0 else { return false }
        guard terms?.allSatisfy(Self.isValid) ?? true,
              spatialTemporal.map(Self.isValid) ?? true else { return false }
        guard let terms else { return true }
        return !terms.isEmpty || spatialTemporal != nil
    }

    private static func isValid(_ term: MultivariateTermSpecification) -> Bool {
        switch term {
        case .spline(let value):
            return value.predictorIndex >= 0 && (0...8).contains(value.knotCount)
        case .tensorProduct(let value):
            return value.firstPredictorIndex >= 0 && value.secondPredictorIndex >= 0
                && value.firstPredictorIndex != value.secondPredictorIndex
                && (0...5).contains(value.firstKnotCount)
                && (0...5).contains(value.secondKnotCount)
        case .categorical(let value):
            guard value.predictorIndex >= 0 else { return false }
            guard let levels = value.levels else { return true }
            // The active execution path later applies its dense or sparse
            // width limit. Keep serialized specifications valid for the
            // bounded high-cardinality sparse contract.
            return (2...1_023).contains(levels.count) && Set(levels).count == levels.count
                && (value.referenceLevel == nil || levels.contains(value.referenceLevel!))
        }
    }

    private static func isValid(_ workflow: SpatialTemporalWorkflowSpecification) -> Bool {
        isValid(.tensorProduct(workflow.spatial))
            && (workflow.temporal.map { isValid(.spline($0)) } ?? true)
    }
}

/// Explicit outcome of a multivariate model fitting attempt.
public enum MultivariateFitStatus: String, Codable, Sendable, Hashable {
    case converged
    case invalidInput
    case numericalFailure
    case lineSearchFailure
    case iterationLimit
}

/// Result that prevents an incomplete IRLS iterate from being used as a model.
public struct MultivariateFitResult: Sendable {
    public let status: MultivariateFitStatus
    public let iterations: Int
    public let deviance: Double?
    public let penalizedObjective: Double?
    public let scoreInfinityNorm: Double?
    /// Non-nil only for `.converged`.
    public let model: MultivariateModel?

    public var converged: Bool { status == .converged && model != nil }

    fileprivate init(
        status: MultivariateFitStatus, iterations: Int, deviance: Double? = nil,
        penalizedObjective: Double? = nil, scoreInfinityNorm: Double? = nil,
        model: MultivariateModel? = nil
    ) {
        self.status = status
        self.iterations = iterations
        self.deviance = deviance
        self.penalizedObjective = penalizedObjective
        self.scoreInfinityNorm = scoreInfinityNorm
        self.model = model
    }
}

/// Conditional covariance and effective degrees of freedom for a multivariate fit.
public struct MultivariateInference: Sendable {
    /// `trace((XᵀWX + 2λPᵀP)⁻¹XᵀWX)` at the final fit.
    public let effectiveDegreesOfFreedom: Double
    /// Exact dense covariance for compact fits. This is empty when
    /// ``method`` is `.diagonalConditionalApproximation`; consumers must use
    /// ``coefficientStandardErrors`` or model-level interval methods instead
    /// of treating a diagonal approximation as a full covariance matrix.
    public let coefficientCovariance: [[Double]]
    public let coefficientStandardErrors: [Double]
    public let method: SparseInferenceMethod
    private let diagonalVariances: [Double]?

    fileprivate init(effectiveDegreesOfFreedom: Double, coefficientCovariance: [[Double]]) {
        self.effectiveDegreesOfFreedom = effectiveDegreesOfFreedom
        self.coefficientCovariance = coefficientCovariance
        coefficientStandardErrors = coefficientCovariance.indices.map {
            sqrt(max(coefficientCovariance[$0][$0], 0))
        }
        method = .exactDenseCovariance
        diagonalVariances = nil
    }

    fileprivate init(effectiveDegreesOfFreedom: Double, diagonalVariances: [Double]) {
        self.effectiveDegreesOfFreedom = effectiveDegreesOfFreedom
        coefficientCovariance = []
        coefficientStandardErrors = diagonalVariances.map { sqrt(max($0, 0)) }
        method = .diagonalConditionalApproximation
        self.diagonalVariances = diagonalVariances
    }

    fileprivate func conditionalVariance(for designRow: [Double]) -> Double? {
        guard designRow.count == coefficientStandardErrors.count else { return nil }
        switch method {
        case .exactDenseCovariance:
            return InferenceMath.quadraticForm(designRow, covariance: coefficientCovariance)
        case .diagonalConditionalApproximation:
            guard let diagonalVariances else { return nil }
            let variance = zip(designRow, diagonalVariances).reduce(0.0) {
                $0 + $1.0 * $1.0 * $1.1
            }
            return variance.isFinite && variance >= 0 ? variance : nil
        }
    }

    fileprivate func scaled(by variance: Double) -> MultivariateInference {
        switch method {
        case .exactDenseCovariance:
            return MultivariateInference(
                effectiveDegreesOfFreedom: effectiveDegreesOfFreedom,
                coefficientCovariance: coefficientCovariance.map { $0.map { $0 * variance } }
            )
        case .diagonalConditionalApproximation:
            return MultivariateInference(
                effectiveDegreesOfFreedom: effectiveDegreesOfFreedom,
                diagonalVariances: (diagonalVariances ?? []).map { $0 * variance }
            )
        }
    }
}

/// One fitted multivariate term with its contribution evaluated on the model scale.
public struct MultivariateTerm: Sendable {
    public let specification: MultivariateTermSpecification
    public let coefficientCount: Int
    fileprivate let basis: MultivariateBasis
    private let coefficients: [Double]

    fileprivate init(specification: MultivariateTermSpecification, basis: MultivariateBasis, coefficients: [Double]) {
        self.specification = specification
        self.basis = basis
        self.coefficients = coefficients
        coefficientCount = coefficients.count
    }

    /// Contribution to the linear predictor, or `nil` for an invalid row/category.
    public func linearPredictorContribution(at x: [Double]) -> Double? {
        basis.values(at: x).map { MultivariateModel.dot($0, coefficients) }
    }

    fileprivate func gradient(at x: [Double], width: Int) -> [Double]? {
        basis.gradient(at: x, coefficients: coefficients, width: width)
    }
}

private enum MultivariateBasis: Sendable {
    case spline(UnivariateBasis)
    case tensor(UnivariateBasis, UnivariateBasis, [Double])
    case categorical(CategoricalBasis)

    func values(at x: [Double]) -> [Double]? {
        switch self {
        case .spline(let basis): return basis.centeredValues(at: x)
        case .tensor(let first, let second, let means):
            guard let left = first.rawValues(at: x), let right = second.rawValues(at: x) else { return nil }
            return left.flatMap { a in right.map { a * $0 } }.enumerated().map { $0.element - means[$0.offset] }
        case .categorical(let basis): return basis.centeredValues(at: x)
        }
    }

    func gradient(at x: [Double], coefficients: [Double], width: Int) -> [Double]? {
        switch self {
        case .spline(let basis):
            guard let derivative = basis.derivatives(at: x) else { return nil }
            var result = [Double](repeating: 0, count: width)
            result[basis.predictorIndex] = MultivariateModel.dot(derivative, coefficients)
            return result
        case .tensor(let first, let second, _):
            guard let left = first.rawValues(at: x), let right = second.rawValues(at: x),
                  let leftDerivative = first.derivatives(at: x), let rightDerivative = second.derivatives(at: x)
            else { return nil }
            var firstGradient = 0.0
            var secondGradient = 0.0
            for firstIndex in left.indices {
                for secondIndex in right.indices {
                    let coefficient = coefficients[firstIndex * right.count + secondIndex]
                    firstGradient += coefficient * leftDerivative[firstIndex] * right[secondIndex]
                    secondGradient += coefficient * left[firstIndex] * rightDerivative[secondIndex]
                }
            }
            var result = [Double](repeating: 0, count: width)
            result[first.predictorIndex] = firstGradient
            result[second.predictorIndex] = secondGradient
            return result
        case .categorical:
            return [Double](repeating: 0, count: width)
        }
    }
}

private struct UnivariateBasis: Sendable {
    let predictorIndex: Int
    let minimum: Double
    let maximum: Double
    let knots: [Double]
    let means: [Double]

    func rawValues(at x: [Double]) -> [Double]? {
        guard x.indices.contains(predictorIndex), x[predictorIndex].isFinite, maximum > minimum else { return nil }
        let u = (x[predictorIndex] - minimum) / (maximum - minimum)
        return [u, u * u, u * u * u] + knots.map { pow(max(u - $0, 0), 3) }
    }

    func centeredValues(at x: [Double]) -> [Double]? {
        rawValues(at: x).map { zip($0, means).map(-) }
    }

    func derivatives(at x: [Double]) -> [Double]? {
        guard x.indices.contains(predictorIndex), x[predictorIndex].isFinite, maximum > minimum else { return nil }
        let range = maximum - minimum
        let u = (x[predictorIndex] - minimum) / range
        return [1 / range, 2 * u / range, 3 * u * u / range] + knots.map {
            3 * pow(max(u - $0, 0), 2) / range
        }
    }
}

private struct CategoricalBasis: Sendable {
    let predictorIndex: Int
    let levels: [Int]
    let referenceLevel: Int

    func centeredValues(at x: [Double]) -> [Double]? {
        guard x.indices.contains(predictorIndex), x[predictorIndex].isFinite,
              let code = exactInteger(x[predictorIndex]), levels.contains(code) else { return nil }
        // Treatment coding is deliberately uncentered: the intercept is the
        // reference-level value and rows retain a truly sparse representation.
        return levels.filter { $0 != referenceLevel }.map { $0 == code ? 1.0 : 0.0 }
    }

    private func exactInteger(_ value: Double) -> Int? {
        guard value.rounded() == value, value >= Double(Int.min), value <= Double(Int.max) else { return nil }
        return Int(value)
    }
}

/// A regular grid of fitted values for a bivariate surface or spatial map.
///
/// `values[yIndex][xIndex]` aligns with `yCoordinates[yIndex]` and
/// `xCoordinates[xIndex]`. Call ``segments(at:)`` for plot-ready marching
/// squares contour segments.
public struct ContourGrid: Codable, Sendable, Hashable {
    public let xPredictorIndex: Int
    public let yPredictorIndex: Int
    public let xCoordinates: [Double]
    public let yCoordinates: [Double]
    public let values: [[Double]]

    public init(
        xPredictorIndex: Int, yPredictorIndex: Int, xCoordinates: [Double],
        yCoordinates: [Double], values: [[Double]]
    ) {
        self.xPredictorIndex = xPredictorIndex
        self.yPredictorIndex = yPredictorIndex
        self.xCoordinates = xCoordinates
        self.yCoordinates = yCoordinates
        self.values = values
    }

    /// Fixed-pairing marching-squares segments at requested response-scale levels.
    ///
    /// Ambiguous saddle cells use a deterministic diagonal pairing; clients that
    /// need topology-aware contour stitching can use `values` directly.
    public func segments(at levels: [Double]) -> [ContourSegment] {
        guard xCoordinates.count >= 2, yCoordinates.count >= 2,
              values.count == yCoordinates.count,
              values.allSatisfy({ $0.count == xCoordinates.count }) else { return [] }
        return levels.filter(\.isFinite).flatMap { level in
            var result: [ContourSegment] = []
            for y in 0..<(yCoordinates.count - 1) {
                for x in 0..<(xCoordinates.count - 1) {
                    let corners = [
                        (xCoordinates[x], yCoordinates[y], values[y][x]),
                        (xCoordinates[x + 1], yCoordinates[y], values[y][x + 1]),
                        (xCoordinates[x + 1], yCoordinates[y + 1], values[y + 1][x + 1]),
                        (xCoordinates[x], yCoordinates[y + 1], values[y + 1][x])
                    ]
                    guard corners.allSatisfy({ $0.2.isFinite }) else { continue }
                    let edges = [(0, 1), (1, 2), (2, 3), (3, 0)].compactMap { edge -> ContourPoint? in
                        let start = corners[edge.0], end = corners[edge.1]
                        guard (start.2 <= level && end.2 > level) || (start.2 > level && end.2 <= level) else {
                            return nil
                        }
                        let fraction = (level - start.2) / (end.2 - start.2)
                        return ContourPoint(
                            x: start.0 + fraction * (end.0 - start.0),
                            y: start.1 + fraction * (end.1 - start.1)
                        )
                    }
                    if edges.count == 2 {
                        result.append(.init(level: level, start: edges[0], end: edges[1]))
                    } else if edges.count == 4 {
                        result.append(.init(level: level, start: edges[0], end: edges[1]))
                        result.append(.init(level: level, start: edges[2], end: edges[3]))
                    }
                }
            }
            return result
        }
    }
}

/// One endpoint in a response-scale contour segment.
public struct ContourPoint: Codable, Sendable, Hashable {
    public let x: Double
    public let y: Double
}

/// One marching-squares contour segment.
public struct ContourSegment: Codable, Sendable, Hashable {
    public let level: Double
    public let start: ContourPoint
    public let end: ContourPoint
}

/// A converged multivariate generalized additive spline model.
public struct MultivariateModel: Sendable {
    public let family: MultivariateResponseFamily
    public let trainX: [[Double]]
    public let trainY: [Double]
    public let intercept: Double
    public let terms: [MultivariateTerm]
    public let fittedValues: [Double]
    public let linearPredictors: [Double]
    public let deviance: Double
    public let nullDeviance: Double
    public let penalizedObjective: Double
    public let penaltyWeight: Double
    /// Numerical solve backend used for the final accepted update.
    public let solverBackend: MultivariateSolverBackend
    /// Native CSR CGLS evidence for the final update, when sparse execution
    /// was accepted. Dense QR and dense fallback retain `nil` here.
    public let sparseExecution: SparseExecutionEvidence?
    public let inference: MultivariateInference
    public let residualScale: Double
    public let iterations: Int
    public let scoreInfinityNorm: Double
    public let keptIndices: [Int]

    private init(
        family: MultivariateResponseFamily, trainX: [[Double]], trainY: [Double],
        intercept: Double, terms: [MultivariateTerm], fittedValues: [Double],
        linearPredictors: [Double], deviance: Double, nullDeviance: Double,
        penalizedObjective: Double, penaltyWeight: Double, inference: MultivariateInference,
        solverBackend: MultivariateSolverBackend, sparseExecution: SparseExecutionEvidence?,
        residualScale: Double, iterations: Int,
        scoreInfinityNorm: Double, keptIndices: [Int]
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
        self.solverBackend = solverBackend
        self.sparseExecution = sparseExecution
        self.inference = inference
        self.residualScale = residualScale
        self.iterations = iterations
        self.scoreInfinityNorm = scoreInfinityNorm
        self.keptIndices = keptIndices
    }

    /// Fit tensor, spline, and categorical terms with penalized WLS or IRLS.
    public static func fit(
        trainX: [[Double]], trainY: [Double], family: MultivariateResponseFamily,
        specification: MultivariateModelSpecification = MultivariateModelSpecification(),
        droppingMissing: Bool = false
    ) -> MultivariateFitResult {
        guard specification.isValid, trainX.count == trainY.count else {
            return .init(status: .invalidInput, iterations: 0)
        }
        let cleaned: ([[Double]], [Double], [Int]) = droppingMissing
            ? MissingData.dropping(trainX: trainX, trainY: trainY)
            : (trainX, trainY, Array(trainX.indices))
        let x = cleaned.0, y = cleaned.1
        guard !x.isEmpty, !x[0].isEmpty, x.allSatisfy({ $0.count == x[0].count }),
              x.flatMap({ $0 }).allSatisfy(\.isFinite), isValidResponse(y, family: family) else {
            return .init(status: .invalidInput, iterations: 0)
        }
        let requested = resolvedTerms(specification: specification, width: x[0].count)
        guard !requested.isEmpty, let basis = makeBasis(
            rows: x, specifications: requested,
            maximumColumns: maximumDesignColumns(for: specification.solverPreference),
            maximumCategoricalLevels: maximumCategoricalLevels(for: specification.solverPreference)
        ) else {
            return .init(status: .invalidInput, iterations: 0)
        }
        let design = basis.design
        let penalty = penaltyRows(columnCount: design[0].count)
        // Build CSR directly from term bases only when this fit may use it.
        // The dense design remains available for exact current inference and
        // diagnostics, but the CGLS operator is no longer rebuilt by scanning
        // that dense matrix at every fit.
        let sparseRequested = sparseRequested(
            design: design, preference: specification.solverPreference
        )
        let nativeSparseDesign = sparseRequested ? sparseDesign(
            rows: x, builders: basis.builders, ranges: basis.ranges,
            columnCount: design[0].count
        ) : nil
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
            let working = eta.map { workingPoint(eta: $0, family: family) }
            let weights = working.map(\.weight)
            let response = eta.indices.map { eta[$0] + (y[$0] - working[$0].mean) / working[$0].derivative }
            guard weights.allSatisfy({ $0.isFinite && $0 > 0 }), response.allSatisfy(\.isFinite),
                  let solve = solvePenalizedWeightedLeastSquares(
                    design: design, response: response, weights: weights, penalty: penalty,
                    penaltyWeight: 2 * specification.penaltyWeight,
                    preference: specification.solverPreference,
                    sparseRequested: sparseRequested, sparseDesign: nativeSparseDesign,
                    sparseMaximumIterations: specification.sparseMaximumIterations,
                    sparseTolerance: specification.sparseTolerance
                  ) else {
                return .init(status: .numericalFailure, iterations: iteration - 1,
                             deviance: current.deviance, penalizedObjective: current.value)
            }
            let proposal = solve.coefficients
            var step = 1.0
            var accepted: (coefficients: [Double], objective: Objective)?
            for _ in 0..<20 {
                let candidate = zip(coefficients, proposal).map { $0 + step * ($1 - $0) }
                let candidateObjective = objective(
                    design: design, response: y, coefficients: candidate,
                    family: family, penaltyWeight: specification.penaltyWeight
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
            let maximumChange = zip(coefficients, accepted.coefficients).map { abs($1 - $0) }.max() ?? 0
            coefficients = accepted.coefficients
            current = accepted.objective
            let score = scoreInfinityNorm(design: design, response: y, coefficients: coefficients,
                                          family: family, penaltyWeight: specification.penaltyWeight)
            guard score.isFinite else {
                return .init(status: .numericalFailure, iterations: iteration,
                             deviance: current.deviance, penalizedObjective: current.value)
            }
            let coefficientScale = max(1, coefficients.map(abs).max() ?? 0)
            let converged = maximumChange <= specification.tolerance * coefficientScale
                && score <= max(specification.tolerance, 1e-7) * scoreScale
            if converged {
                let linearPredictors = design.map { dot($0, coefficients) }
                let fitted = linearPredictors.map { mean(eta: $0, family: family) }
                let unitScaleInference: MultivariateInference?
                if solve.backend == .sparseCGLS && design[0].count > denseInferenceColumnLimit {
                    unitScaleInference = makeDiagonalInference(
                        design: design, linearPredictors: linearPredictors, family: family,
                        penaltyWeight: specification.penaltyWeight, residualScale: 1
                    )
                } else {
                    unitScaleInference = makeInference(
                        design: design, linearPredictors: linearPredictors, family: family,
                        penaltyWeight: specification.penaltyWeight, residualScale: 1
                    )
                }
                guard let unitScaleInference else {
                    return .init(status: .numericalFailure, iterations: iteration,
                                 deviance: current.deviance, penalizedObjective: current.value,
                                 scoreInfinityNorm: score)
                }
                let residualScale = family == .gaussian
                    ? sqrt(current.deviance / max(Double(y.count) - unitScaleInference.effectiveDegreesOfFreedom, 1)) : 1
                let inference: MultivariateInference
                if family == .gaussian {
                    let variance = residualScale * residualScale
                    inference = unitScaleInference.scaled(by: variance)
                } else {
                    inference = unitScaleInference
                }
                let terms = basis.builders.enumerated().map { offset, builder in
                    MultivariateTerm(
                        specification: builder.specification, basis: builder.basis,
                        coefficients: Array(coefficients[basis.ranges[offset]])
                    )
                }
                let model = MultivariateModel(
                    family: family, trainX: x, trainY: y, intercept: coefficients[0], terms: terms,
                    fittedValues: fitted, linearPredictors: linearPredictors,
                    deviance: current.deviance, nullDeviance: nullDeviance(y, family: family),
                    penalizedObjective: current.value, penaltyWeight: specification.penaltyWeight,
                    inference: inference, solverBackend: solve.backend,
                    sparseExecution: solve.sparseExecution,
                    residualScale: residualScale, iterations: iteration,
                    scoreInfinityNorm: score, keptIndices: cleaned.2
                )
                return .init(status: .converged, iterations: iteration, deviance: current.deviance,
                             penalizedObjective: current.value, scoreInfinityNorm: score, model: model)
            }
        }
        return .init(status: .iterationLimit, iterations: specification.maxIterations,
                     deviance: current.deviance, penalizedObjective: current.value)
    }

    /// Fitted response mean at a complete predictor row.
    public func predict(_ x: [Double]) -> Double {
        guard let eta = linearPredictor(x) else { return .nan }
        return Self.mean(eta: eta, family: family)
    }

    /// Batch response means.
    public func predict(_ xs: [[Double]]) -> [Double] { xs.map(predict) }

    /// Linear predictor at a complete predictor row.
    public func linearPredictor(_ x: [Double]) -> Double? {
        guard x.count == trainX[0].count, x.allSatisfy(\.isFinite) else { return nil }
        var result = intercept
        for term in terms {
            guard let contribution = term.linearPredictorContribution(at: x) else { return nil }
            result += contribution
        }
        return result
    }

    /// Gradient of the fitted response mean in original predictor coordinates.
    public func gradient(at x: [Double]) -> [Double]? {
        guard let eta = linearPredictor(x) else { return nil }
        var result = [Double](repeating: 0, count: x.count)
        for term in terms {
            guard let gradient = term.gradient(at: x, width: x.count) else { return nil }
            for index in result.indices { result[index] += gradient[index] }
        }
        let multiplier = Self.workingPoint(eta: eta, family: family).derivative
        return result.map { multiplier * $0 }
    }

    /// Conditional response-mean standard error, fixed terms and penalty.
    public func standardError(at x: [Double]) -> Double? {
        guard let eta = linearPredictor(x), let row = designRow(for: x),
              let variance = inference.conditionalVariance(for: row)
        else { return nil }
        return Self.workingPoint(eta: eta, family: family).derivative * sqrt(variance)
    }

    /// Conditional normal-approximation response-mean interval.
    public func meanConfidenceInterval(
        at x: [Double], confidenceLevel: Double = 0.95
    ) -> StatisticalInterval? {
        guard confidenceLevel.isFinite, confidenceLevel > 0, confidenceLevel < 1,
              let eta = linearPredictor(x), let row = designRow(for: x),
              let variance = inference.conditionalVariance(for: row),
              let z = InferenceMath.normalQuantile(0.5 + confidenceLevel / 2) else { return nil }
        let error = sqrt(variance)
        return StatisticalInterval(
            estimate: Self.mean(eta: eta, family: family),
            lowerBound: Self.mean(eta: eta - z * error, family: family),
            upperBound: Self.mean(eta: eta + z * error, family: family),
            confidenceLevel: confidenceLevel
        )
    }

    /// Family-correct residuals aligned with `keptIndices`.
    public func residuals(_ kind: ResidualKind) -> [Double] {
        zip(trainY, fittedValues).map { y, fitted in
            let raw = y - fitted
            switch kind {
            case .raw: return raw
            case .pearson:
                switch family {
                case .gaussian: return residualScale > 0 ? raw / residualScale : (raw == 0 ? 0 : .nan)
                case .binomial: return raw / sqrt(max(fitted * (1 - fitted), 1e-12))
                case .poisson: return raw / sqrt(max(fitted, 1e-12))
                }
            case .deviance:
                if family == .gaussian { return raw }
                let sign = raw < 0 ? -1.0 : raw > 0 ? 1.0 : 0
                return sign * sqrt(max(Self.unitDeviance(y: y, mean: fitted, family: family), 0))
            }
        }
    }

    /// Plot-ready surface over two predictors with all other values held at `baseline`.
    public func contour(
        xPredictorIndex: Int, yPredictorIndex: Int, baseline: [Double],
        xCount: Int = 50, yCount: Int = 50
    ) -> ContourGrid? {
        guard xCount >= 2, yCount >= 2, xPredictorIndex != yPredictorIndex,
              baseline.count == trainX[0].count, baseline.allSatisfy(\.isFinite),
              trainX[0].indices.contains(xPredictorIndex), trainX[0].indices.contains(yPredictorIndex),
              let xMinimum = trainX.map({ $0[xPredictorIndex] }).min(),
              let xMaximum = trainX.map({ $0[xPredictorIndex] }).max(), xMaximum > xMinimum,
              let yMinimum = trainX.map({ $0[yPredictorIndex] }).min(),
              let yMaximum = trainX.map({ $0[yPredictorIndex] }).max(), yMaximum > yMinimum else { return nil }
        let xs = (0..<xCount).map { xMinimum + (xMaximum - xMinimum) * Double($0) / Double(xCount - 1) }
        let ys = (0..<yCount).map { yMinimum + (yMaximum - yMinimum) * Double($0) / Double(yCount - 1) }
        let values = ys.map { yValue in
            xs.map { xValue in
                var point = baseline
                point[xPredictorIndex] = xValue
                point[yPredictorIndex] = yValue
                return predict(point)
            }
        }
        guard values.flatMap({ $0 }).allSatisfy(\.isFinite) else { return nil }
        return ContourGrid(
            xPredictorIndex: xPredictorIndex, yPredictorIndex: yPredictorIndex,
            xCoordinates: xs, yCoordinates: ys, values: values
        )
    }

    /// Diagnostics compatible with the unified fitted-model contract.
    public var diagnostics: FitDiagnostics {
        FitDiagnostics(
            responseFamily: family.responseFamily,
            linkFunction: family == .gaussian ? .identity : family == .binomial ? .logit : .log,
            observationCount: trainY.count, effectiveDegreesOfFreedom: inference.effectiveDegreesOfFreedom,
            residualScale: residualScale, deviance: deviance, nullDeviance: nullDeviance
        )
    }

    fileprivate static func dot(_ left: [Double], _ right: [Double]) -> Double {
        zip(left, right).reduce(0) { $0 + $1.0 * $1.1 }
    }

    private struct Objective {
        let value: Double
        let deviance: Double
        var isFinite: Bool { value.isFinite && deviance.isFinite }
    }

    /// Full fixed-basis covariance is cubic in the number of coefficients.
    /// Above this threshold only accepted CSR fits receive the documented
    /// diagonal conditional approximation.
    private static let denseInferenceColumnLimit = 256
    /// This phase deliberately keeps the temporary dense diagnostic design
    /// bounded while allowing high-cardinality categorical sparse workflows.
    private static let sparseDesignColumnLimit = 1_024

    /// A term-native CSR design. Its values are emitted directly by each
    /// fitted basis, rather than derived by scanning the dense diagnostic
    /// design. This lightweight representation keeps the DataLens core
    /// portable; conversion to `NumericCoreSparse.SparseMatrix` happens only
    /// at the Swift-NumericCore execution boundary.
    private struct NativeSparseDesign: Sendable {
        let rows: Int
        let columns: Int
        let rowPointers: [Int]
        let columnIndices: [Int]
        let values: [Double]

        var nonZeroCount: Int { values.count }
    }

    private struct NumericalSolve {
        let coefficients: [Double]
        let backend: MultivariateSolverBackend
        let sparseExecution: SparseExecutionEvidence?
    }

    private struct BasisBuilder {
        let specification: MultivariateTermSpecification
        let basis: MultivariateBasis
    }

    /// Emit CSR rows from the fitted bases. Categorical treatment terms add
    /// just their active indicator; this avoids the former dense-to-CSR
    /// re-encoding path for the large-factor workflows that select CGLS.
    private static func sparseDesign(
        rows: [[Double]], builders: [BasisBuilder], ranges: [Range<Int>],
        columnCount: Int
    ) -> NativeSparseDesign? {
        guard !rows.isEmpty, columnCount > 0, builders.count == ranges.count else { return nil }
        var rowPointers = [0]
        var columnIndices: [Int] = []
        var values: [Double] = []
        for row in rows {
            columnIndices.append(0)
            values.append(1)
            for (builder, range) in zip(builders, ranges) {
                guard let termValues = builder.basis.values(at: row), termValues.count == range.count else {
                    return nil
                }
                for offset in termValues.indices where termValues[offset] != 0 {
                    columnIndices.append(range.lowerBound + offset)
                    values.append(termValues[offset])
                }
            }
            rowPointers.append(values.count)
        }
        return NativeSparseDesign(
            rows: rows.count, columns: columnCount, rowPointers: rowPointers,
            columnIndices: columnIndices, values: values
        )
    }

    private static func resolvedTerms(
        specification: MultivariateModelSpecification, width: Int
    ) -> [MultivariateTermSpecification] {
        var values = specification.terms ?? (0..<width).map {
            .spline(SplineTermSpecification(predictorIndex: $0, knotCount: specification.defaultKnotCount))
        }
        if let workflow = specification.spatialTemporal {
            let firstMain = MultivariateTermSpecification.spline(.init(
                predictorIndex: workflow.spatial.firstPredictorIndex,
                knotCount: workflow.spatial.firstKnotCount
            ))
            let secondMain = MultivariateTermSpecification.spline(.init(
                predictorIndex: workflow.spatial.secondPredictorIndex,
                knotCount: workflow.spatial.secondKnotCount
            ))
            if !values.contains(firstMain) { values.append(firstMain) }
            if !values.contains(secondMain) { values.append(secondMain) }
            let spatialTensor = MultivariateTermSpecification.tensorProduct(workflow.spatial)
            if !values.contains(spatialTensor) { values.append(spatialTensor) }
            if let temporal = workflow.temporal {
                let temporalTerm = MultivariateTermSpecification.spline(temporal)
                if !values.contains(temporalTerm) { values.append(temporalTerm) }
            }
        }
        return values
    }

    private static func maximumDesignColumns(for preference: MultivariateSolverPreference) -> Int {
        switch preference {
        case .denseQR:
            return denseInferenceColumnLimit
        case .automatic, .sparseCGLS:
            return sparseDesignColumnLimit
        }
    }

    private static func maximumCategoricalLevels(for preference: MultivariateSolverPreference) -> Int {
        // Treatment coding omits one reference level. Reserve a small amount
        // for an intercept and other terms inside the design-width contract.
        max(2, maximumDesignColumns(for: preference) - 1)
    }

    private static func makeBasis(
        rows: [[Double]], specifications: [MultivariateTermSpecification],
        maximumColumns: Int, maximumCategoricalLevels: Int
    ) -> (design: [[Double]], builders: [BasisBuilder], ranges: [Range<Int>])? {
        var columns = [[Double](repeating: 1, count: rows.count)]
        var builders: [BasisBuilder] = []
        var ranges: [Range<Int>] = []
        for specification in specifications {
            let basis: MultivariateBasis
            let values: [[Double]]
            switch specification {
            case .spline(let value):
                guard let univariate = makeUnivariate(rows: rows, predictorIndex: value.predictorIndex,
                                                      knotCount: value.knotCount) else { return nil }
                basis = .spline(univariate)
                guard let termValues = rows.map({ univariate.centeredValues(at: $0) }).all() else { return nil }
                values = termValues
            case .tensorProduct(let value):
                guard let first = makeUnivariate(rows: rows, predictorIndex: value.firstPredictorIndex,
                                                 knotCount: value.firstKnotCount),
                      let second = makeUnivariate(rows: rows, predictorIndex: value.secondPredictorIndex,
                                                  knotCount: value.secondKnotCount),
                      let raw = rows.map({ row -> [Double]? in
                          guard let left = first.rawValues(at: row), let right = second.rawValues(at: row) else { return nil }
                          return left.flatMap { a in right.map { a * $0 } }
                      }).all() else { return nil }
                let means = raw[0].indices.map { index in raw.map { $0[index] }.reduce(0, +) / Double(rows.count) }
                basis = .tensor(first, second, means)
                values = raw.map { zip($0, means).map(-) }
            case .categorical(let value):
                guard let categorical = makeCategorical(
                    rows: rows, specification: value,
                    maximumLevels: maximumCategoricalLevels
                ),
                      let termValues = rows.map({ categorical.centeredValues(at: $0) }).all() else { return nil }
                basis = .categorical(categorical)
                values = termValues
            }
            guard !values.isEmpty, !values[0].isEmpty, values.allSatisfy({ $0.count == values[0].count }) else {
                return nil
            }
            let start = columns.count
            for index in values[0].indices { columns.append(values.map { $0[index] }) }
            ranges.append(start..<columns.count)
            builders.append(.init(specification: specification, basis: basis))
        }
        let design = rows.indices.map { row in columns.map { $0[row] } }
        guard design[0].count <= maximumColumns else { return nil }
        return (design, builders, ranges)
    }

    private static func makeUnivariate(
        rows: [[Double]], predictorIndex: Int, knotCount: Int
    ) -> UnivariateBasis? {
        guard rows.first?.indices.contains(predictorIndex) == true, (0...8).contains(knotCount) else { return nil }
        let values = rows.map { $0[predictorIndex] }
        guard let minimum = values.min(), let maximum = values.max(), maximum > minimum else { return nil }
        let knots = knotCount == 0 ? [] : (1...knotCount).map {
            Double($0) / Double(knotCount + 1)
        }
        let raw = values.map { value -> [Double] in
            let u = (value - minimum) / (maximum - minimum)
            return [u, u * u, u * u * u] + knots.map { pow(max(u - $0, 0), 3) }
        }
        let means = raw[0].indices.map { index in raw.map { $0[index] }.reduce(0, +) / Double(rows.count) }
        return UnivariateBasis(
            predictorIndex: predictorIndex, minimum: minimum, maximum: maximum, knots: knots, means: means
        )
    }

    private static func makeCategorical(
        rows: [[Double]], specification: CategoricalTermSpecification,
        maximumLevels: Int
    ) -> CategoricalBasis? {
        guard rows.first?.indices.contains(specification.predictorIndex) == true else { return nil }
        let codes = rows.compactMap { exactInteger($0[specification.predictorIndex]) }
        guard codes.count == rows.count else { return nil }
        let levels = specification.levels ?? Array(Set(codes)).sorted()
        guard (2...maximumLevels).contains(levels.count), Set(levels).count == levels.count,
              codes.allSatisfy(levels.contains) else { return nil }
        let reference = specification.referenceLevel ?? levels[0]
        guard levels.contains(reference) else { return nil }
        return CategoricalBasis(
            predictorIndex: specification.predictorIndex, levels: levels, referenceLevel: reference
        )
    }

    private static func exactInteger(_ value: Double) -> Int? {
        guard value.rounded() == value, value >= Double(Int.min), value <= Double(Int.max) else { return nil }
        return Int(value)
    }

    private static func isValidResponse(_ y: [Double], family: MultivariateResponseFamily) -> Bool {
        switch family {
        case .gaussian: y.allSatisfy(\.isFinite)
        case .binomial: y.allSatisfy { $0 == 0 || $0 == 1 }
        case .poisson: y.allSatisfy { $0.isFinite && $0 >= 0 }
        }
    }

    private static func mean(eta: Double, family: MultivariateResponseFamily) -> Double {
        switch family {
        case .gaussian: return eta
        case .binomial:
            let value = min(max(eta, -30), 30)
            return value >= 0 ? 1 / (1 + exp(-value)) : exp(value) / (1 + exp(value))
        case .poisson: return max(exp(min(max(eta, -30), 30)), 1e-12)
        }
    }

    private static func workingPoint(eta: Double, family: MultivariateResponseFamily) -> (mean: Double, derivative: Double, weight: Double) {
        let mean = mean(eta: eta, family: family)
        switch family {
        case .gaussian: return (mean, 1, 1)
        case .binomial:
            let derivative = max(mean * (1 - mean), 1e-12)
            return (mean, derivative, derivative)
        case .poisson: return (mean, mean, mean)
        }
    }

    private static func startIntercept(_ y: [Double], family: MultivariateResponseFamily) -> Double {
        let mean = y.reduce(0, +) / Double(y.count)
        switch family {
        case .gaussian: return mean
        case .binomial:
            let probability = min(max(mean, 1e-6), 1 - 1e-6)
            return log(probability / (1 - probability))
        case .poisson: return log(max(mean, 1e-6))
        }
    }

    private static func objective(
        design: [[Double]], response: [Double], coefficients: [Double],
        family: MultivariateResponseFamily, penaltyWeight: Double
    ) -> Objective {
        let deviance = zip(design, response).reduce(0.0) {
            $0 + unitDeviance(y: $1.1, mean: mean(eta: dot($1.0, coefficients), family: family), family: family)
        }
        let penalty = coefficients.dropFirst().reduce(0.0) { $0 + $1 * $1 }
        return Objective(value: 0.5 * deviance + penaltyWeight * penalty, deviance: deviance)
    }

    private static func nullDeviance(_ y: [Double], family: MultivariateResponseFamily) -> Double {
        let fitted = mean(eta: startIntercept(y, family: family), family: family)
        return y.reduce(0) { $0 + unitDeviance(y: $1, mean: fitted, family: family) }
    }

    private static func unitDeviance(y: Double, mean: Double, family: MultivariateResponseFamily) -> Double {
        switch family {
        case .gaussian: return pow(y - mean, 2)
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
        family: MultivariateResponseFamily, penaltyWeight: Double
    ) -> Double {
        var score = [Double](repeating: 0, count: coefficients.count)
        for (row, y) in zip(design, response) {
            let residual = y - mean(eta: dot(row, coefficients), family: family)
            for index in score.indices { score[index] += row[index] * residual }
        }
        for index in score.indices.dropFirst() { score[index] -= 2 * penaltyWeight * coefficients[index] }
        return score.map(abs).max() ?? .infinity
    }

    private func designRow(for x: [Double]) -> [Double]? {
        guard x.count == trainX[0].count, x.allSatisfy(\.isFinite) else { return nil }
        var row = [1.0]
        for term in terms {
            guard let values = term.basis.values(at: x) else { return nil }
            row += values
        }
        return row
    }

    private static func penaltyRows(columnCount: Int) -> [[Double]] {
        (1..<columnCount).map { index in
            var row = [Double](repeating: 0, count: columnCount)
            row[index] = 1
            return row
        }
    }

    private static func solvePenalizedWeightedLeastSquares(
        design: [[Double]], response: [Double], weights: [Double], penalty: [[Double]], penaltyWeight: Double,
        preference: MultivariateSolverPreference, sparseRequested: Bool,
        sparseDesign: NativeSparseDesign?, sparseMaximumIterations: Int,
        sparseTolerance: Double
    ) -> NumericalSolve? {
        if sparseRequested {
            #if canImport(NumericCoreSparse)
            if let sparseDesign, let sparse = sparsePenalizedSolve(
                design: sparseDesign, response: response, weights: weights, penaltyWeight: penaltyWeight,
                maximumIterations: sparseMaximumIterations, tolerance: sparseTolerance
            ) {
                return sparse
            }
            guard preference == .automatic else { return nil }
            #else
            guard preference == .automatic else { return nil }
            #endif
        }
        #if canImport(NumericCoreAccelerate)
        guard let coefficients = StatisticalSolver.penalizedWeightedLeastSquares(
            design: design, response: response, weights: weights,
            penaltyRows: penalty, penaltyWeight: penaltyWeight
        )?.coefficients else { return nil }
        return NumericalSolve(
            coefficients: coefficients,
            backend: sparseRequested ? .denseQRFallback : .denseQR,
            sparseExecution: nil
        )
        #else
        var augmented: [[Double]] = []
        var augmentedResponse: [Double] = []
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
        guard let coefficients = LinAlg.leastSquares(design: augmented, response: augmentedResponse) else {
            return nil
        }
        return NumericalSolve(
            coefficients: coefficients,
            backend: sparseRequested ? .denseQRFallback : .denseQR,
            sparseExecution: nil
        )
        #endif
    }

    private static func sparseRequested(
        design: [[Double]], preference: MultivariateSolverPreference
    ) -> Bool {
        switch preference {
        case .automatic: return sparseGeometryIsWorthwhile(design)
        case .denseQR: return false
        case .sparseCGLS: return true
        }
    }

    /// Phase 14 dispatch rule, calibrated against the release benchmark's
    /// n=12,000/p=253 categorical workload. Dense QR remains faster and more
    /// accurate for compact or substantially dense spline designs.
    private static func sparseGeometryIsWorthwhile(_ design: [[Double]]) -> Bool {
        guard let columnCount = design.first?.count, columnCount > 0,
              design.count * columnCount >= 250_000 else { return false }
        // QR grows with both row count and the square of the 253-column factor
        // design; past this point low-density factor models consistently favor
        // the portable CGLS path in the release workload.
        let nonzeros = design.reduce(0) { count, row in
            count + row.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
        }
        return Double(nonzeros) / Double(design.count * columnCount) <= 0.12
    }

    #if canImport(NumericCoreSparse)
    private static func sparsePenalizedSolve(
        design: NativeSparseDesign, response: [Double], weights: [Double], penaltyWeight: Double,
        maximumIterations: Int, tolerance: Double
    ) -> NumericalSolve? {
        guard let sparseDesign = try? SparseMatrix<Double>(
            rows: design.rows, cols: design.columns, rowPointers: design.rowPointers,
            columnIndices: design.columnIndices, values: design.values
        ), let sparsePenalty = identityPenalty(columnCount: design.columns) else { return nil }
        do {
            let result = try SparseStatisticalSolver.penalizedWeightedLeastSquares(
                design: sparseDesign, response: response, weights: weights, penalty: sparsePenalty,
                penaltyWeight: penaltyWeight,
                maxIterations: maximumIterations, tolerance: tolerance
            )
            guard result.converged, result.coefficients.count == design.columns,
                  result.coefficients.allSatisfy(\.isFinite) else { return nil }
            return NumericalSolve(
                coefficients: result.coefficients, backend: .sparseCGLS,
                sparseExecution: SparseExecutionEvidence(
                    designRows: design.rows, designColumns: design.columns,
                    nonZeroCount: design.nonZeroCount, iterations: result.iterations,
                    normalResidualNorm: result.residualNorm, converged: result.converged,
                    weightedResidualSumOfSquares: result.weightedResidualSumOfSquares,
                    penaltyContribution: result.penaltyContribution,
                    workingSolveIterationLimit: maximumIterations,
                    workingSolveTolerance: tolerance,
                    inferenceMethod: design.columns > denseInferenceColumnLimit
                        ? .diagonalConditionalApproximation : .exactDenseCovariance
                )
            )
        } catch {
            return nil
        }
    }

    private static func identityPenalty(columnCount: Int) -> SparseMatrix<Double>? {
        guard columnCount > 1 else { return nil }
        let columns = Array(1..<columnCount)
        return try? SparseMatrix(
            rows: columnCount - 1, cols: columnCount,
            rowPointers: Array(0..<columnCount), columnIndices: columns,
            values: [Double](repeating: 1, count: columnCount - 1)
        )
    }
    #endif

    private static func makeInference(
        design: [[Double]], linearPredictors: [Double], family: MultivariateResponseFamily,
        penaltyWeight: Double, residualScale: Double
    ) -> MultivariateInference? {
        guard !design.isEmpty, design.count == linearPredictors.count,
              let count = design.first?.count, count > 0 else { return nil }
        var information = Array(repeating: [Double](repeating: 0, count: count), count: count)
        for rowIndex in design.indices {
            let weight = workingPoint(eta: linearPredictors[rowIndex], family: family).weight
            guard weight.isFinite && weight > 0 else { return nil }
            for left in 0..<count {
                for right in left..<count {
                    information[left][right] += weight * design[rowIndex][left] * design[rowIndex][right]
                }
            }
        }
        for row in information.indices {
            for column in 0..<row { information[row][column] = information[column][row] }
        }
        var penalized = information
        for index in penalized.indices.dropFirst() { penalized[index][index] += 2 * penaltyWeight }
        var inverseColumns: [[Double]] = []
        for column in 0..<count {
            var unit = [Double](repeating: 0, count: count)
            unit[column] = 1
            guard let solution = Regression.solveSPD(penalized, unit), solution.allSatisfy(\.isFinite) else { return nil }
            inverseColumns.append(solution)
        }
        let inverse = (0..<count).map { row in inverseColumns.map { $0[row] } }
        var hat = Array(repeating: [Double](repeating: 0, count: count), count: count)
        for row in 0..<count {
            for column in 0..<count {
                for inner in 0..<count { hat[row][column] += inverse[row][inner] * information[inner][column] }
            }
        }
        var covariance = Array(repeating: [Double](repeating: 0, count: count), count: count)
        for row in 0..<count {
            for column in 0..<count {
                for inner in 0..<count { covariance[row][column] += hat[row][inner] * inverse[inner][column] }
            }
        }
        if family == .gaussian {
            let variance = residualScale * residualScale
            for row in covariance.indices {
                for column in covariance[row].indices { covariance[row][column] *= variance }
            }
        }
        for row in 0..<count {
            for column in 0..<row {
                let symmetric = 0.5 * (covariance[row][column] + covariance[column][row])
                covariance[row][column] = symmetric
                covariance[column][row] = symmetric
            }
        }
        let edf = hat.indices.reduce(0.0) { $0 + hat[$1][$1] }
        guard edf.isFinite, edf >= 1 - 1e-8, edf <= Double(count) + 1e-8,
              covariance.flatMap({ $0 }).allSatisfy(\.isFinite) else { return nil }
        return MultivariateInference(
            effectiveDegreesOfFreedom: min(max(edf, 1), Double(count)), coefficientCovariance: covariance
        )
    }

    /// Bounded conditional-inference approximation for large accepted CSR
    /// fits. It keeps only `diag(XᵀWX + 2λPᵀP)⁻¹`, so it does not assert that
    /// cross-coefficient covariance is zero; it is explicitly a diagonal
    /// approximation used to keep uncertainty and EDF available without an
    /// O(p³) factorization or O(p²) retained covariance.
    private static func makeDiagonalInference(
        design: [[Double]], linearPredictors: [Double], family: MultivariateResponseFamily,
        penaltyWeight: Double, residualScale: Double
    ) -> MultivariateInference? {
        guard !design.isEmpty, design.count == linearPredictors.count,
              let count = design.first?.count, count > denseInferenceColumnLimit,
              design.allSatisfy({ $0.count == count }) else { return nil }
        var informationDiagonal = [Double](repeating: 0, count: count)
        for rowIndex in design.indices {
            let weight = workingPoint(eta: linearPredictors[rowIndex], family: family).weight
            guard weight.isFinite && weight > 0 else { return nil }
            for index in design[rowIndex].indices {
                let value = design[rowIndex][index]
                informationDiagonal[index] += weight * value * value
            }
        }
        var diagonalVariances = [Double](repeating: 0, count: count)
        var edf = 0.0
        for index in informationDiagonal.indices {
            let information = informationDiagonal[index]
            let penalized = information + (index == 0 ? 0 : 2 * penaltyWeight)
            guard information.isFinite, penalized.isFinite, penalized > 0 else { return nil }
            diagonalVariances[index] = residualScale * residualScale / penalized
            edf += information / penalized
        }
        guard edf.isFinite, edf >= 1 - 1e-8, edf <= Double(count) + 1e-8,
              diagonalVariances.allSatisfy({ $0.isFinite && $0 >= 0 }) else { return nil }
        return MultivariateInference(
            effectiveDegreesOfFreedom: min(max(edf, 1), Double(count)),
            diagonalVariances: diagonalVariances
        )
    }
}

private extension Array where Element == [Double]? {
    func all() -> [[Double]]? {
        guard allSatisfy({ $0 != nil }) else { return nil }
        return compactMap { $0 }
    }
}
