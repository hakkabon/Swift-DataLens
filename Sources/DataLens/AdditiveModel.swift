import Foundation

/// Configuration for one smooth term in an ``AdditiveModel``.
///
/// Terms currently use one-dimensional LOESS. Keeping the predictor index in
/// the specification permits selected-variable models and different smoothing
/// strengths per predictor without expanding the input matrix.
public struct AdditiveTermSpecification: Sendable, Hashable {
    public let predictorIndex: Int
    public let span: Double
    public let degree: Int

    public init(predictorIndex: Int, span: Double = 0.75, degree: Int = 2) {
        self.predictorIndex = predictorIndex
        self.span = span
        self.degree = degree
    }
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
