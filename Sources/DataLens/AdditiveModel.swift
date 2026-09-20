import Foundation

/// Configuration for one smooth term in an ``AdditiveModel``.
///
/// Terms currently use one-dimensional LOESS. Keeping the predictor index in
/// the specification permits selected-variable models and different smoothing
/// strengths per predictor without expanding the input matrix.
public struct AdditiveTermSpecification: Codable, Sendable, Hashable {
    public let predictorIndex: Int
    public let span: Double
    public let degree: Int

    public init(predictorIndex: Int, span: Double = 0.75, degree: Int = 2) {
        self.predictorIndex = predictorIndex
        self.span = span
        self.degree = degree
    }
}

/// Reproducible configuration for a Gaussian additive main-effects model.
///
/// Supplying `terms` permits a selected-variable model with per-term smoothness;
/// `nil` creates one term for every predictor. The remaining values mirror
/// ``AdditiveModel/fit(trainX:trainY:terms:defaultSpan:defaultDegree:robustIterations:maxIterations:tolerance:droppingMissing:)``
/// so a fit can be replayed by a workbench or validation run without UI state.
public struct AdditiveModelSpecification: Codable, Sendable, Hashable {
    public let terms: [AdditiveTermSpecification]?
    public let defaultSpan: Double
    public let defaultDegree: Int
    public let robustIterations: Int
    public let maxIterations: Int
    public let tolerance: Double

    public init(
        terms: [AdditiveTermSpecification]? = nil,
        defaultSpan: Double = 0.75, defaultDegree: Int = 2,
        robustIterations: Int = 0, maxIterations: Int = 100,
        tolerance: Double = 1e-8
    ) {
        self.terms = terms
        self.defaultSpan = defaultSpan
        self.defaultDegree = defaultDegree
        self.robustIterations = robustIterations
        self.maxIterations = maxIterations
        self.tolerance = tolerance
    }

    var isValid: Bool {
        guard defaultSpan.isFinite && defaultSpan > 0 && defaultSpan <= 1,
              (0...2).contains(defaultDegree),
              robustIterations >= 0, maxIterations > 0,
              tolerance.isFinite && tolerance > 0 else { return false }
        guard let terms else { return true }
        return !terms.isEmpty && terms.allSatisfy {
            $0.span.isFinite && $0.span > 0 && $0.span <= 1
                && (0...2).contains($0.degree) && $0.predictorIndex >= 0
        }
    }
}

/// An interpretable summary of one fitted additive term.
///
/// The effect is centered over the retained training rows, so `meanEffect`
/// is numerically close to zero and the model intercept retains its ordinary
/// response-scale interpretation.
public struct AdditiveTermDiagnostics: Sendable, Hashable {
    public let predictorIndex: Int
    public let effectiveDegreesOfFreedom: Double
    public let meanEffect: Double
    public let rootMeanSquareEffect: Double
    public let minimumEffect: Double
    public let maximumEffect: Double
}

/// A regular, centered partial-effect curve for one additive term.
///
/// `gradient` remains aligned with `x`; unavailable derivatives are `NaN`.
/// This keeps a visual client from mistaking a missing derivative for zero.
public struct AdditivePartialEffect: Sendable, Hashable {
    public let predictorIndex: Int
    public let x: [Double]
    public let effect: [Double]
    public let gradient: [Double]
}

/// A centered, fitted smooth term from an ``AdditiveModel``.
public struct AdditiveTerm: Sendable {
    public let specification: AdditiveTermSpecification
    public let smoother: Loess
    /// Mean raw smoother value removed to make the term identifiable.
    public let centeringOffset: Double

    /// The centered contribution of this term at a predictor value.
    public func predict(_ value: Double,
                        extrapolation: ExtrapolationPolicy = .polynomial) -> Double {
        smoother.predict([value], extrapolation: extrapolation) - centeringOffset
    }

    /// Centered contributions for a batch of predictor values.
    public func predict(_ values: [Double],
                        extrapolation: ExtrapolationPolicy = .polynomial) -> [Double] {
        smoother.predict(values.map { [$0] }, extrapolation: extrapolation)
            .map { $0 - centeringOffset }
    }

    /// Derivative of this term, or nil for a degree-zero/degenerate fit.
    public func gradient(at value: Double) -> Double? {
        smoother.gradient(at: [value])?.first
    }

    /// Approximate effective degrees of freedom contributed by this centered term.
    public var effectiveDegreesOfFreedom: Double { max(smoother.trace - 1, 0) }

    /// Effect-size diagnostics on the retained rows used to fit this term.
    public var diagnostics: AdditiveTermDiagnostics {
        let effects = predict(smoother.trainX.map { $0[0] })
        let n = Double(max(effects.count, 1))
        let mean = effects.reduce(0, +) / n
        let rms = sqrt(effects.reduce(0) { $0 + $1 * $1 } / n)
        return AdditiveTermDiagnostics(
            predictorIndex: specification.predictorIndex,
            effectiveDegreesOfFreedom: effectiveDegreesOfFreedom,
            meanEffect: mean, rootMeanSquareEffect: rms,
            minimumEffect: effects.min() ?? .nan, maximumEffect: effects.max() ?? .nan
        )
    }

    /// Regular partial-effect curve over this term's retained predictor hull.
    ///
    /// The returned effect excludes the model intercept and every other term.
    /// It is therefore suitable for a component plot, not a response-scale
    /// prediction plot.
    public func partialEffect(count: Int = 100,
                              extrapolation: ExtrapolationPolicy = .polynomial) -> AdditivePartialEffect? {
        guard count >= 2,
              let low = smoother.trainX.map({ $0[0] }).min(),
              let high = smoother.trainX.map({ $0[0] }).max(), high > low else { return nil }
        let x = (0..<count).map { low + (high - low) * Double($0) / Double(count - 1) }
        return AdditivePartialEffect(
            predictorIndex: specification.predictorIndex,
            x: x, effect: predict(x, extrapolation: extrapolation),
            gradient: x.map { gradient(at: $0) ?? .nan }
        )
    }
}

/// A Gaussian generalized additive model fitted by classical backfitting.
///
/// The model is `y = intercept + Σ fⱼ(xⱼ)`. Every component is centered over
/// the training rows, making the intercept and component effects identifiable.
/// Fits that do not satisfy the convergence tolerance before `maxIterations`
/// return nil; an unconverged model is never exposed as a successful result.
///
/// This first additive layer intentionally models main effects only. Interactions,
/// categorical terms, likelihood families, and joint uncertainty are future API
/// extensions rather than implicit approximations.
public struct AdditiveModel: Sendable {
    public let trainX: [[Double]]
    public let trainY: [Double]
    public let intercept: Double
    public let terms: [AdditiveTerm]
    public let fittedValues: [Double]
    public let sigma: Double
    /// Approximate effective degrees of freedom: intercept plus centered term traces.
    public let effectiveDegreesOfFreedom: Double
    public let iterations: Int
    public let maximumChange: Double
    public let keptIndices: [Int]

    /// Interpretable per-term effect summaries over the retained training rows.
    public var termDiagnostics: [AdditiveTermDiagnostics] { terms.map(\.diagnostics) }

    /// A component curve for the requested predictor, or `nil` when that
    /// predictor was excluded or its observed range is degenerate.
    public func partialEffect(
        forPredictor predictorIndex: Int, count: Int = 100,
        extrapolation: ExtrapolationPolicy = .polynomial
    ) -> AdditivePartialEffect? {
        terms.first(where: { $0.specification.predictorIndex == predictorIndex })?
            .partialEffect(count: count, extrapolation: extrapolation)
    }

    private init(trainX: [[Double]], trainY: [Double], intercept: Double,
                 terms: [AdditiveTerm], fittedValues: [Double], sigma: Double,
                 effectiveDegreesOfFreedom: Double, iterations: Int,
                 maximumChange: Double, keptIndices: [Int]) {
        self.trainX = trainX
        self.trainY = trainY
        self.intercept = intercept
        self.terms = terms
        self.fittedValues = fittedValues
        self.sigma = sigma
        self.effectiveDegreesOfFreedom = effectiveDegreesOfFreedom
        self.iterations = iterations
        self.maximumChange = maximumChange
        self.keptIndices = keptIndices
    }

    /// Fit additive main effects with cyclic backfitting.
    ///
    /// When `terms` is nil, every predictor receives a LOESS term using
    /// `defaultSpan` and `defaultDegree`. Robust iterations are disabled by
    /// default because ordinary linear smoothers have the clearest backfitting
    /// convergence behavior; they remain opt-in for contaminated data.
    public static func fit(trainX: [[Double]], trainY: [Double],
                           terms requestedTerms: [AdditiveTermSpecification]? = nil,
                           defaultSpan: Double = 0.75, defaultDegree: Int = 2,
                           robustIterations: Int = 0,
                           maxIterations: Int = 100, tolerance: Double = 1e-8,
                           droppingMissing: Bool = false) -> AdditiveModel? {
        guard trainX.count == trainY.count else { return nil }
        let cleaned: ([[Double]], [Double], [Int]) = droppingMissing
            ? MissingData.dropping(trainX: trainX, trainY: trainY)
            : (trainX, trainY, Array(trainX.indices))
        let x = cleaned.0
        let y = cleaned.1
        guard !x.isEmpty, !x[0].isEmpty,
              x.allSatisfy({ $0.count == x[0].count }),
              x.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              y.allSatisfy({ $0.isFinite }),
              defaultSpan > 0, defaultSpan <= 1, (0...2).contains(defaultDegree),
              robustIterations >= 0, maxIterations > 0,
              tolerance > 0, tolerance.isFinite else { return nil }

        let specifications = requestedTerms ?? x[0].indices.map {
            AdditiveTermSpecification(predictorIndex: $0,
                                      span: defaultSpan, degree: defaultDegree)
        }
        guard !specifications.isEmpty,
              Set(specifications.map(\.predictorIndex)).count == specifications.count,
              specifications.allSatisfy({
                  x[0].indices.contains($0.predictorIndex)
                    && $0.span > 0 && $0.span <= 1 && (0...2).contains($0.degree)
              }) else { return nil }

        let n = y.count
        let intercept = y.reduce(0, +) / Double(n)
        var contributions = specifications.map { _ in [Double](repeating: 0, count: n) }
        var fittedTerms = [AdditiveTerm?](repeating: nil, count: specifications.count)
        let responseScale = max(1, y.map { abs($0 - intercept) }.max() ?? 0)
        var finalChange = Double.infinity
        var completedIterations = 0

        for iteration in 1...maxIterations {
            var iterationChange = 0.0
            for j in specifications.indices {
                let partial = y.indices.map { i in
                    var value = y[i] - intercept
                    for k in contributions.indices where k != j { value -= contributions[k][i] }
                    return value
                }
                let spec = specifications[j]
                let column = x.map { [$0[spec.predictorIndex]] }
                guard let smoother = Loess.fit(trainX: column, trainY: partial,
                                               span: spec.span, degree: spec.degree,
                                               robustIterations: robustIterations) else { return nil }
                let raw = smoother.fittedValues
                let offset = raw.reduce(0, +) / Double(n)
                let centered = raw.map { $0 - offset }
                for i in centered.indices {
                    iterationChange = max(iterationChange, abs(centered[i] - contributions[j][i]))
                }
                contributions[j] = centered
                fittedTerms[j] = AdditiveTerm(specification: spec, smoother: smoother,
                                              centeringOffset: offset)
            }
            completedIterations = iteration
            finalChange = iterationChange
            if iterationChange <= tolerance * responseScale { break }
        }
        guard finalChange <= tolerance * responseScale,
              fittedTerms.allSatisfy({ $0 != nil }) else { return nil }
        let terms = fittedTerms.compactMap { $0 }
        let fitted = y.indices.map { i in
            intercept + contributions.reduce(0) { $0 + $1[i] }
        }
        let edf = 1 + terms.reduce(0) { $0 + max($1.smoother.trace - 1, 0) }
        let rss = zip(y, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma = sqrt(rss / max(Double(n) - edf, 1))
        return AdditiveModel(trainX: x, trainY: y, intercept: intercept,
                             terms: terms, fittedValues: fitted, sigma: sigma,
                             effectiveDegreesOfFreedom: edf,
                             iterations: completedIterations, maximumChange: finalChange,
                             keptIndices: cleaned.2)
    }

    /// Centered contribution from each term at a complete predictor row.
    public func componentContributions(at x: [Double],
                                       extrapolation: ExtrapolationPolicy = .polynomial) -> [Double]? {
        guard x.count == trainX[0].count, x.allSatisfy({ $0.isFinite }) else { return nil }
        return terms.map { term in
            term.predict(x[term.specification.predictorIndex], extrapolation: extrapolation)
        }
    }

    /// Predict a response for a complete predictor row.
    public func predict(_ x: [Double],
                        extrapolation: ExtrapolationPolicy = .polynomial) -> Double {
        guard let effects = componentContributions(at: x, extrapolation: extrapolation) else {
            return .nan
        }
        return intercept + effects.reduce(0, +)
    }

    /// Predict responses for multiple complete predictor rows.
    public func predict(_ xs: [[Double]],
                        extrapolation: ExtrapolationPolicy = .polynomial) -> [Double] {
        xs.map { predict($0, extrapolation: extrapolation) }
    }

    /// Additive gradient in original predictor coordinates.
    ///
    /// Predictors without a term have derivative zero. Nil means at least one
    /// fitted term cannot provide a derivative at the query (for example a
    /// degree-zero term).
    public func gradient(at x: [Double]) -> [Double]? {
        guard x.count == trainX[0].count, x.allSatisfy({ $0.isFinite }) else { return nil }
        var result = [Double](repeating: 0, count: x.count)
        for term in terms {
            let index = term.specification.predictorIndex
            guard let derivative = term.gradient(at: x[index]) else { return nil }
            result[index] = derivative
        }
        return result
    }
}
