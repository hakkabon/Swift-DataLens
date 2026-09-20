import Foundation

/// The fitting strategy used by a unified statistical-model specification.
///
/// Automatic smoothing retains its family-routing behavior. Additive fitting
/// is deliberately explicit: a caller never gets a likelihood GAM merely
/// because a response happens to contain integers.
public enum StatisticalModelStrategy: String, Codable, Sendable, Hashable {
    case automaticSmoothing
    case additiveGaussian
    case additiveBinomial
    case additivePoisson
}

/// Serializable configuration for a reproducible statistical fit.
///
/// This mirrors the public automatic-smoother arguments rather than retaining
/// a UI-specific budget. It can therefore be used by an application, a batch
/// job, or ``CrossValidation`` with the same meaning.
public struct StatisticalModelSpecification: Codable, Sendable, Hashable {
    public let strategy: StatisticalModelStrategy
    public let degree: Int
    public let spans: [Double]?
    public let robustIterations: Int
    public let adaptiveContender: Bool
    /// Configuration used when `strategy` is `.additiveGaussian`.
    /// `nil` means the default main-effects GAM over every predictor.
    public let additive: AdditiveModelSpecification?
    /// Configuration used by `.additiveBinomial` and `.additivePoisson`.
    /// `nil` selects the default centered regression-spline basis and penalty.
    public let likelihoodAdditive: LikelihoodAdditiveModelSpecification?

    public init(
        strategy: StatisticalModelStrategy = .automaticSmoothing,
        degree: Int = 2, spans: [Double]? = nil, robustIterations: Int = 4,
        adaptiveContender: Bool = true,
        additive: AdditiveModelSpecification? = nil,
        likelihoodAdditive: LikelihoodAdditiveModelSpecification? = nil
    ) {
        self.strategy = strategy
        self.degree = degree
        self.spans = spans
        self.robustIterations = robustIterations
        self.adaptiveContender = adaptiveContender
        self.additive = additive
        self.likelihoodAdditive = likelihoodAdditive
    }

    var isValid: Bool {
        switch strategy {
        case .automaticSmoothing:
            (0...2).contains(degree)
                && robustIterations >= 0
                && (spans == nil || (!(spans?.isEmpty ?? true) && spans!.allSatisfy {
                    $0.isFinite && $0 > 0 && $0 <= 1
                }))
        case .additiveGaussian:
            additive?.isValid ?? true
        case .additiveBinomial, .additivePoisson:
            likelihoodAdditive?.isValid ?? true
        }
    }
}

/// The concrete family of a unified fitted model.
public enum StatisticalModelKind: String, Codable, Sendable, Hashable {
    case smoother
    case additiveGaussian
    case additiveBinomial
    case additivePoisson
}

/// A common fitted-model contract for smoothers and additive main-effects.
///
/// It records retained training rows, model-family diagnostics, predictions,
/// gradients, and documented residual definitions in one place. The wrapper
/// does not manufacture uncertainty: additive main effects currently return
/// `nil` from ``standardError(at:extrapolation:)`` until joint uncertainty is
/// implemented.
public struct FittedStatisticalModel: Sendable {
    private enum Storage: Sendable {
        case smoother(FittedSmoother)
        case additive(AdditiveModel)
        case likelihoodAdditive(LikelihoodAdditiveModel)
    }

    private let storage: Storage
    public let kind: StatisticalModelKind
    public let specification: StatisticalModelSpecification?
    public let tuningSummary: TuningSummary?
    /// Retained predictor rows, parallel to ``trainingResponses``.
    public let trainingPredictors: [[Double]]
    /// Retained responses, parallel to ``trainingPredictors``.
    public let trainingResponses: [Double]
    /// Original input-row positions for the retained training rows.
    public let keptIndices: [Int]
    public let diagnostics: FitDiagnostics

    /// Wrap an existing smoother and recover its retained training rows from
    /// the caller's original arrays. Returns `nil` when those arrays cannot
    /// support the smoother's recorded join-back indices.
    public init?(
        smoother: FittedSmoother, trainX: [[Double]], trainY: [Double],
        specification: StatisticalModelSpecification? = nil,
        tuningSummary: TuningSummary? = nil
    ) {
        guard trainX.count == trainY.count,
              specification?.isValid ?? true,
              smoother.keptIndices.allSatisfy({ trainX.indices.contains($0) && trainY.indices.contains($0) })
        else { return nil }
        let rows = smoother.keptIndices
        let retainedX = rows.map { trainX[$0] }
        let retainedY = rows.map { trainY[$0] }
        guard retainedX.count == smoother.fittedValues.count,
              retainedX.first.map({ row in retainedX.allSatisfy { $0.count == row.count } }) ?? false
        else { return nil }
        storage = .smoother(smoother)
        kind = .smoother
        self.specification = specification
        self.tuningSummary = tuningSummary
        trainingPredictors = retainedX
        trainingResponses = retainedY
        keptIndices = rows
        diagnostics = smoother.diagnostics
    }

    /// Wrap a converged Gaussian additive main-effects model.
    public init(
        additive: AdditiveModel,
        specification: StatisticalModelSpecification? = nil
    ) {
        storage = .additive(additive)
        kind = .additiveGaussian
        self.specification = specification
        tuningSummary = nil
        trainingPredictors = additive.trainX
        trainingResponses = additive.trainY
        keptIndices = additive.keptIndices
        let residuals = zip(additive.trainY, additive.fittedValues).map { $0 - $1 }
        let deviance = residuals.reduce(0.0) { $0 + $1 * $1 }
        let mean = additive.trainY.reduce(0, +) / Double(max(additive.trainY.count, 1))
        let nullDeviance = additive.trainY.reduce(0.0) { $0 + pow($1 - mean, 2) }
        diagnostics = FitDiagnostics(
            responseFamily: .gaussian, linkFunction: .identity,
            observationCount: additive.trainY.count,
            effectiveDegreesOfFreedom: additive.effectiveDegreesOfFreedom,
            residualScale: additive.sigma, deviance: deviance,
            nullDeviance: nullDeviance
        )
    }

    /// Wrap a converged binomial or Poisson penalized additive model.
    public init(
        likelihoodAdditive: LikelihoodAdditiveModel,
        specification: StatisticalModelSpecification? = nil
    ) {
        storage = .likelihoodAdditive(likelihoodAdditive)
        kind = likelihoodAdditive.family == .binomial ? .additiveBinomial : .additivePoisson
        self.specification = specification
        tuningSummary = nil
        trainingPredictors = likelihoodAdditive.trainX
        trainingResponses = likelihoodAdditive.trainY
        keptIndices = likelihoodAdditive.keptIndices
        diagnostics = likelihoodAdditive.diagnostics
    }

    /// Fit the configured model and expose it through the common contract.
    /// Missing rows are deliberately dropped so retained indices are always
    /// meaningful to reports and validation consumers.
    public static func fit(
        trainX: [[Double]], trainY: [Double],
        specification: StatisticalModelSpecification = StatisticalModelSpecification()
    ) -> FittedStatisticalModel? {
        guard specification.isValid else { return nil }
        switch specification.strategy {
        case .automaticSmoothing:
            guard let result = AutomaticSmoother.fit(
                trainX: trainX, trainY: trainY, degree: specification.degree,
                spans: specification.spans, robustIterations: specification.robustIterations,
                droppingMissing: true, adaptiveContender: specification.adaptiveContender
            ) else { return nil }
            return FittedStatisticalModel(
                smoother: result.fit, trainX: trainX, trainY: trainY,
                specification: specification, tuningSummary: result.summary
            )
        case .additiveGaussian:
            let additiveSpecification = specification.additive ?? AdditiveModelSpecification()
            guard let additive = AdditiveModel.fit(
                trainX: trainX, trainY: trainY, terms: additiveSpecification.terms,
                defaultSpan: additiveSpecification.defaultSpan,
                defaultDegree: additiveSpecification.defaultDegree,
                robustIterations: additiveSpecification.robustIterations,
                maxIterations: additiveSpecification.maxIterations,
                tolerance: additiveSpecification.tolerance,
                droppingMissing: true
            ) else { return nil }
            return FittedStatisticalModel(additive: additive, specification: specification)
        case .additiveBinomial, .additivePoisson:
            let family: LikelihoodAdditiveFamily = specification.strategy == .additiveBinomial
                ? .binomial : .poisson
            let likelihoodSpecification = specification.likelihoodAdditive
                ?? LikelihoodAdditiveModelSpecification()
            guard let additive = LikelihoodAdditiveModel.fit(
                trainX: trainX, trainY: trainY, family: family,
                specification: likelihoodSpecification, droppingMissing: true
            ).model else { return nil }
            return FittedStatisticalModel(
                likelihoodAdditive: additive, specification: specification
            )
        }
    }

    /// Training fitted values, parallel to ``trainingResponses``.
    public var fittedValues: [Double] {
        switch storage {
        case .smoother(let smoother): smoother.fittedValues
        case .additive(let additive): additive.fittedValues
        case .likelihoodAdditive(let additive): additive.fittedValues
        }
    }

    /// Fitted mean at a predictor row, or `.nan` when it cannot be evaluated.
    public func predict(_ x: [Double], extrapolation: ExtrapolationPolicy = .polynomial) -> Double {
        switch storage {
        case .smoother(let smoother): smoother.predict(x, extrapolation: extrapolation)
        case .additive(let additive): additive.predict(x, extrapolation: extrapolation)
        case .likelihoodAdditive(let additive): additive.predict(x)
        }
    }

    /// Batch fitted means.
    public func predict(_ xs: [[Double]], extrapolation: ExtrapolationPolicy = .polynomial) -> [Double] {
        switch storage {
        case .smoother(let smoother): smoother.predict(xs, extrapolation: extrapolation)
        case .additive(let additive): additive.predict(xs, extrapolation: extrapolation)
        case .likelihoodAdditive(let additive): additive.predict(xs)
        }
    }

    /// Gradient in original predictor coordinates.
    public func gradient(at x: [Double]) -> [Double]? {
        switch storage {
        case .smoother(let smoother): smoother.gradient(at: x)
        case .additive(let additive): additive.gradient(at: x)
        case .likelihoodAdditive(let additive): additive.gradient(at: x)
        }
    }

    /// Conditional mean standard error where the model supplies one.
    public func standardError(
        at x: [Double], extrapolation: ExtrapolationPolicy = .polynomial
    ) -> Double? {
        switch storage {
        case .smoother(let smoother): return smoother.standardError(at: x, extrapolation: extrapolation)
        case .additive, .likelihoodAdditive: return nil
        }
    }

    /// Interpretable summaries for fitted GAM terms.
    ///
    /// This is empty for smoother-backed models, so consumers that render a
    /// generic model report can use one property without downcasting storage.
    public var additiveTermDiagnostics: [AdditiveTermDiagnostics] {
        switch storage {
        case .smoother: []
        case .additive(let additive): additive.termDiagnostics
        case .likelihoodAdditive: []
        }
    }

    /// Centered component curve for one GAM predictor.
    ///
    /// Returns `nil` for non-additive models, excluded predictors, or a
    /// degenerate observed predictor range. The curve is a term contribution,
    /// not a full response-scale prediction.
    public func partialEffect(
        forPredictor predictorIndex: Int, count: Int = 100,
        extrapolation: ExtrapolationPolicy = .polynomial
    ) -> AdditivePartialEffect? {
        switch storage {
        case .smoother: nil
        case .additive(let additive):
            additive.partialEffect(
                forPredictor: predictorIndex, count: count, extrapolation: extrapolation
            )
        case .likelihoodAdditive(let additive):
            additive.partialEffect(forPredictor: predictorIndex, count: count)
        }
    }

    /// Family-correct retained-training residuals.
    public func residuals(_ kind: ResidualKind) -> [Double] {
        switch storage {
        case .smoother(let smoother): return smoother.residuals(kind)
        case .additive:
            let raw = zip(trainingResponses, fittedValues).map { $0 - $1 }
            switch kind {
            case .raw, .deviance: return raw
            case .pearson:
                let scale = diagnostics.residualScale
                return raw.map { scale > 0 ? $0 / scale : ($0 == 0 ? 0 : .nan) }
            }
        case .likelihoodAdditive(let additive):
            return additive.residuals(kind)
        }
    }
}
