import Foundation

/// A fitted smoother of unknown-ahead-of-time kind, as returned by
/// `AutomaticSmoother` — or wrapped explicitly for uniform evaluation
/// (`.nadarayaWatson` and `.whittakerEilers` and `.totalVariation`:
/// the tuner never produces them, keeping kernel, penalized, and
/// edge-preserving selection explicit per the app track's decisions).
public enum FittedSmoother: Sendable {
    case loess(Loess)
    case adaptive(AdaptiveLoess)
    case likelihood(LocalLikelihood)
    case nadarayaWatson(NadarayaWatson)
    case whittakerEilers(WhittakerEilers)
    case totalVariation(TotalVariation)

    /// Fitted values at the training points.
    public var fittedValues: [Double] {
        switch self {
        case .loess(let fit): fit.fittedValues
        case .adaptive(let fit): fit.fittedValues
        case .likelihood(let fit): fit.fittedValues
        case .nadarayaWatson(let fit): fit.fittedValues
        case .whittakerEilers(let fit): fit.fittedValues
        case .totalVariation(let fit): fit.fittedValues
        }
    }

    /// Predict at `x` (`.nan` on width mismatch, mirroring each smoother).
    public func predict(_ x: [Double], extrapolation: ExtrapolationPolicy = .polynomial) -> Double {
        switch self {
        case .loess(let fit): fit.predict(x, extrapolation: extrapolation)
        case .adaptive(let fit): fit.predict(x, extrapolation: extrapolation)
        case .likelihood(let fit): fit.predict(x, extrapolation: extrapolation)
        case .nadarayaWatson(let fit): fit.predict(x, extrapolation: extrapolation)
        case .whittakerEilers(let fit): fit.predict(x, extrapolation: extrapolation)
        case .totalVariation(let fit): fit.predict(x, extrapolation: extrapolation)
        }
    }

    /// Predictions over many queries.
    public func predict(_ xs: [[Double]], extrapolation: ExtrapolationPolicy = .polynomial) -> [Double] {
        switch self {
        case .loess(let fit): fit.predict(xs, extrapolation: extrapolation)
        case .adaptive(let fit): fit.predict(xs, extrapolation: extrapolation)
        case .likelihood(let fit): fit.predict(xs, extrapolation: extrapolation)
        case .nadarayaWatson(let fit): fit.predict(xs, extrapolation: extrapolation)
        case .whittakerEilers(let fit): fit.predict(xs, extrapolation: extrapolation)
        case .totalVariation(let fit): fit.predict(xs, extrapolation: extrapolation)
        }
    }

    /// Concurrent batch predictions (identical to `predict(_:)`).
    public func predictConcurrently(_ xs: [[Double]],
                                    extrapolation: ExtrapolationPolicy = .polynomial) async throws -> [Double] {
        switch self {
        case .loess(let fit): try await fit.predictConcurrently(xs, extrapolation: extrapolation)
        case .adaptive(let fit): try await fit.predictConcurrently(xs, extrapolation: extrapolation)
        case .likelihood(let fit): try await fit.predictConcurrently(xs, extrapolation: extrapolation)
        case .nadarayaWatson(let fit): try await fit.predictConcurrently(xs, extrapolation: extrapolation)
        case .whittakerEilers(let fit): try await fit.predictConcurrently(xs, extrapolation: extrapolation)
        case .totalVariation(let fit): try await fit.predictConcurrently(xs, extrapolation: extrapolation)
        }
    }

    /// Gradient ∇ŷ(x) (nil where the underlying smoother says nil).
    public func gradient(at x: [Double]) -> [Double]? {
        switch self {
        case .loess(let fit): fit.gradient(at: x)
        case .adaptive(let fit): fit.gradient(at: x)
        case .likelihood(let fit): fit.gradient(at: x)
        case .nadarayaWatson(let fit): fit.gradient(at: x)
        case .whittakerEilers(let fit): fit.gradient(at: x)
        case .totalVariation(let fit): fit.gradient(at: x)
        }
    }

    /// Gradients over many queries.
    public func gradients(at xs: [[Double]]) -> [[Double]?] {
        switch self {
        case .loess(let fit): fit.gradients(at: xs)
        case .adaptive(let fit): fit.gradients(at: xs)
        case .likelihood(let fit): fit.gradients(at: xs)
        case .nadarayaWatson(let fit): fit.gradients(at: xs)
        case .whittakerEilers(let fit): fit.gradients(at: xs)
        case .totalVariation(let fit): fit.gradients(at: xs)
        }
    }

    /// Concurrent batch gradients (identical to `gradients(at:)`).
    public func gradientsConcurrently(at xs: [[Double]]) async throws -> [[Double]?] {
        switch self {
        case .loess(let fit): try await fit.gradientsConcurrently(at: xs)
        case .adaptive(let fit): try await fit.gradientsConcurrently(at: xs)
        case .likelihood(let fit): try await fit.gradientsConcurrently(at: xs)
        case .nadarayaWatson(let fit): try await fit.gradientsConcurrently(at: xs)
        case .whittakerEilers(let fit): try await fit.gradientsConcurrently(at: xs)
        case .totalVariation(let fit): try await fit.gradientsConcurrently(at: xs)
        }
    }

    /// Kept input row indices (identity when nothing was dropped).
    public var keptIndices: [Int] {
        switch self {
        case .loess(let fit): fit.keptIndices
        case .adaptive(let fit): fit.keptIndices
        case .likelihood(let fit): fit.keptIndices
        case .nadarayaWatson(let fit): fit.keptIndices
        case .whittakerEilers(let fit): fit.keptIndices
        case .totalVariation(let fit): fit.keptIndices
        }
    }

    /// Standard error at `x` (nil where the underlying smoother says nil).
    public func standardError(at x: [Double],
                              extrapolation: ExtrapolationPolicy = .polynomial) -> Double? {
        switch self {
        case .loess(let fit): fit.standardError(at: x, extrapolation: extrapolation)
        case .adaptive(let fit): fit.standardError(at: x, extrapolation: extrapolation)
        case .likelihood(let fit): fit.standardError(at: x, extrapolation: extrapolation)
        case .nadarayaWatson(let fit): fit.standardError(at: x, extrapolation: extrapolation)
        case .whittakerEilers(let fit): fit.standardError(at: x, extrapolation: extrapolation)
        case .totalVariation(let fit): fit.standardError(at: x, extrapolation: extrapolation)
        }
    }

    /// Standard errors over many queries.
    public func standardErrors(at xs: [[Double]],
                               extrapolation: ExtrapolationPolicy = .polynomial) -> [Double?] {
        switch self {
        case .loess(let fit): fit.standardErrors(at: xs, extrapolation: extrapolation)
        case .adaptive(let fit): fit.standardErrors(at: xs, extrapolation: extrapolation)
        case .likelihood(let fit): fit.standardErrors(at: xs, extrapolation: extrapolation)
        case .nadarayaWatson(let fit): fit.standardErrors(at: xs, extrapolation: extrapolation)
        case .whittakerEilers(let fit): fit.standardErrors(at: xs, extrapolation: extrapolation)
        case .totalVariation(let fit): fit.standardErrors(at: xs, extrapolation: extrapolation)
        }
    }

    /// Concurrent batch standard errors (identical to `standardErrors(at:)`).
    public func standardErrorsConcurrently(at xs: [[Double]],
                                           extrapolation: ExtrapolationPolicy = .polynomial) async throws -> [Double?] {
        switch self {
        case .loess(let fit): try await fit.standardErrorsConcurrently(at: xs, extrapolation: extrapolation)
        case .adaptive(let fit): try await fit.standardErrorsConcurrently(at: xs, extrapolation: extrapolation)
        case .likelihood(let fit): try await fit.standardErrorsConcurrently(at: xs, extrapolation: extrapolation)
        case .nadarayaWatson(let fit): try await fit.standardErrorsConcurrently(at: xs, extrapolation: extrapolation)
        case .whittakerEilers(let fit): try await fit.standardErrorsConcurrently(at: xs, extrapolation: extrapolation)
        case .totalVariation(let fit): try await fit.standardErrorsConcurrently(at: xs, extrapolation: extrapolation)
        }
    }
}

/// Human-readable record of what `AutomaticSmoother` chose and why.
public struct TuningSummary: Sendable, CustomStringConvertible {
    /// Chosen smoother, e.g. "AdaptiveLoess".
    public let smoother: String
    /// How it was configured, e.g. "span 0.5 (AIC 61.96)".
    public let detail: String
    /// Winning criterion value (GCV or AIC, lower is better).
    public let score: Double
    /// Why, in one or two sentences.
    public let reason: String
    /// Fallbacks and caveats (empty in the common case).
    public let notes: [String]

    public var description: String {
        var lines = ["AutomaticSmoother", "  Smoother: \(smoother) (\(detail))",
                     "  Criterion score: \(score)", "  Reason: \(reason)"]
        for note in notes { lines.append("  Note: \(note)") }
        return lines.joined(separator: "\n")
    }
}

/// Automated tuning in one call: inspect the responses, route to a smoother
/// family, and select its smoothing parameter by an information criterion.
///
/// Routing: all-0/1 responses → binomial local likelihood; non-negative
/// integral responses → Poisson local likelihood; anything else →
/// continuous smoothers, where `AdaptiveLoess` (per-point AICc) and
/// fixed-span `Loess` compete on GCV and the winner is kept. Span
/// candidates for the fixed-span legs default to `[0.3, 0.5, 0.75]`.
///
/// A preferred path that fails degrades gracefully (recorded in
/// `TuningSummary.notes`): tiny inputs where `AdaptiveLoess` has no valid
/// neighborhood fall back to fixed-span `Loess`; a failed likelihood path
/// falls back to the continuous smoothers. Total failure returns nil.
public enum AutomaticSmoother {
    /// Default span candidates for the fixed-span legs.
    public static let defaultSpans = [0.3, 0.5, 0.75]

    enum DataKind {
        case binary
        case counts
        case continuous
    }

    static func classify(_ trainY: [Double]) -> DataKind {
        if trainY.allSatisfy({ $0 == 0 || $0 == 1 }) { return .binary }
        if trainY.allSatisfy({ $0 >= 0 && $0 == $0.rounded() }) { return .counts }
        return .continuous
    }

    /// GCV score from fitted values and smoother trace (mirrors
    /// `Loess.selectSpan`'s formula so fixed and adaptive spans compare).
    static func gcv(fitted: [Double], trainY: [Double], trace: Double) -> Double {
        let n = Double(trainY.count)
        let rss = zip(trainY, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let denom = max(1 - trace / n, 1e-6)
        return (rss / n) / (denom * denom)
    }

    /// Tune the continuous smoothers: `AdaptiveLoess` against the best
    /// fixed-span `Loess` on GCV. Returns the winner, its GCV, and a detail
    /// string. Notes adaptive unavailability (tiny inputs) for the summary.
    /// With `adaptiveContender: false` the adaptive leg is skipped entirely
    /// (shallow tuning for interactive use) and only fixed spans compete.
    static func tuneContinuous(trainX: [[Double]], trainY: [Double], degree: Int,
                               spans: [Double], robustIterations: Int, droppingMissing: Bool,
                               adaptiveContender: Bool = true,
                               notes: inout [String]) -> (fit: FittedSmoother, score: Double, detail: String)? {
        var best: (fit: FittedSmoother, score: Double, detail: String)?
        if adaptiveContender {
            if let adaptive = AdaptiveLoess.fit(trainX: trainX, trainY: trainY, degree: degree,
                                                robustIterations: robustIterations,
                                                droppingMissing: droppingMissing) {
                // GCV on the fit's own (possibly dropped) rows.
                let score = gcv(fitted: adaptive.fittedValues, trainY: adaptive.trainY, trace: adaptive.trace)
                best = (.adaptive(adaptive), score, "per-point AICc neighborhoods (GCV \(score))")
            } else {
                notes.append("AdaptiveLoess has no valid neighborhood here; comparing fixed spans only.")
            }
        } else {
            notes.append("Adaptive contender disabled (shallow tuning); comparing fixed spans only.")
        }
        if let (_, loess) = Loess.selectSpan(trainX: trainX, trainY: trainY, spans: spans,
                                             degree: degree, robustIterations: robustIterations,
                                             droppingMissing: droppingMissing) {
            let score = gcv(fitted: loess.fittedValues, trainY: loess.trainY, trace: loess.trace)
            let detail = "span \(loess.span) (GCV \(score))"
            if let current = best {
                if score < current.score { best = (.loess(loess), score, detail) }
            } else {
                best = (.loess(loess), score, detail)
            }
        }
        return best
    }

    /// Tune one likelihood family by AIC (deviance + 2·trace).
    static func tuneLikelihood(trainX: [[Double]], trainY: [Double], degree: Int,
                               family: LocalLikelihoodFamily, droppingMissing: Bool,
                               spans: [Double]) -> (fit: FittedSmoother, score: Double, detail: String)? {
        guard let (_, fit) = LocalLikelihood.selectSpan(trainX: trainX, trainY: trainY, spans: spans,
                                                        degree: degree, family: family,
                                                        droppingMissing: droppingMissing) else { return nil }
        let score = fit.deviance + 2 * fit.trace
        let name: String
        switch family {
        case .binomial: name = "Binomial local likelihood"
        case .poisson: name = "Poisson local likelihood"
        case .gaussian: name = "Gaussian local likelihood"
        }
        return (.likelihood(fit), score, "\(name), span \(fit.span) (AIC \(score))")
    }

    /// Fit automatically: route by response type, tune, and report.
    ///
    /// With `droppingMissing`, classification inspects the finite responses
    /// and the legs drop non-finite rows (their `keptIndices` stay correct).
    /// With `adaptiveContender: false`, the adaptive leg is skipped and
    /// only fixed-span `Loess` competes (shallow tuning for interactive
    /// use); the default `true` preserves the full competition.
    public static func fit(trainX: [[Double]], trainY: [Double], degree: Int = 2,
                           spans: [Double]? = nil,
                           robustIterations: Int = 4,
                           droppingMissing: Bool = false,
                           adaptiveContender: Bool = true) -> (fit: FittedSmoother, summary: TuningSummary)? {
        guard trainX.count == trainY.count, (0...2).contains(degree), !trainX.isEmpty else { return nil }
        let classY = droppingMissing ? trainY.filter({ $0.isFinite }) : trainY
        guard !classY.isEmpty else { return nil }
        let spans = spans ?? defaultSpans
        var notes: [String] = []
        switch classify(classY) {
        case .binary:
            if let (fit, score, detail) = tuneLikelihood(trainX: trainX, trainY: trainY, degree: degree,
                                                        family: .binomial, droppingMissing: droppingMissing,
                                                        spans: spans) {
                let summary = TuningSummary(
                    smoother: "LocalLikelihood", detail: detail, score: score,
                    reason: "Binary responses route to binomial local likelihood; span selected by AIC.",
                    notes: notes
                )
                return (fit, summary)
            }
            notes.append("Binomial tuning failed; falling back to continuous smoothers.")
            return tunedContinuousFallback(trainX: trainX, trainY: trainY, degree: degree,
                                           spans: spans, robustIterations: robustIterations,
                                           droppingMissing: droppingMissing,
                                           adaptiveContender: adaptiveContender, notes: notes)
        case .counts:
            if let (fit, score, detail) = tuneLikelihood(trainX: trainX, trainY: trainY, degree: degree,
                                                        family: .poisson, droppingMissing: droppingMissing,
                                                        spans: spans) {
                let summary = TuningSummary(
                    smoother: "LocalLikelihood", detail: detail, score: score,
                    reason: "Non-negative integer responses route to Poisson local likelihood; span selected by AIC.",
                    notes: notes
                )
                return (fit, summary)
            }
            notes.append("Poisson tuning failed; falling back to continuous smoothers.")
            return tunedContinuousFallback(trainX: trainX, trainY: trainY, degree: degree,
                                           spans: spans, robustIterations: robustIterations,
                                           droppingMissing: droppingMissing,
                                           adaptiveContender: adaptiveContender, notes: notes)
        case .continuous:
            return tunedContinuousFallback(trainX: trainX, trainY: trainY, degree: degree,
                                           spans: spans, robustIterations: robustIterations,
                                           droppingMissing: droppingMissing,
                                           adaptiveContender: adaptiveContender, notes: notes)
        }
    }

    private static func tunedContinuousFallback(trainX: [[Double]], trainY: [Double], degree: Int,
                                                spans: [Double], robustIterations: Int, droppingMissing: Bool,
                                                adaptiveContender: Bool = true,
                                                notes: [String]) -> (fit: FittedSmoother, summary: TuningSummary)? {
        var notes = notes
        guard let (fit, score, detail) = tuneContinuous(trainX: trainX, trainY: trainY, degree: degree,
                                                       spans: spans, robustIterations: robustIterations,
                                                       droppingMissing: droppingMissing,
                                                       adaptiveContender: adaptiveContender,
                                                       notes: &notes) else { return nil }
        let smoother: String
        let reason: String
        switch fit {
        case .adaptive:
            smoother = "AdaptiveLoess"
            reason = "Continuous responses; per-point AICc neighborhoods beat fixed-span Loess on GCV."
        case .loess:
            smoother = "Loess"
            reason = adaptiveContender
                ? "Continuous responses; fixed-span Loess beat AdaptiveLoess on GCV."
                : "Continuous responses; fixed-span Loess selected by GCV (adaptive contender disabled)."
        case .likelihood:
            smoother = "LocalLikelihood"
            reason = "Continuous responses; likelihood path selected."
        case .nadarayaWatson:
            // Unreachable through tuneContinuous (the tuner never routes
            // kernel fits); present so the switch stays exhaustive if an
            // explicit fit ever flows here.
            smoother = "NadarayaWatson"
            reason = "Continuous responses; kernel fit selected explicitly."
        case .whittakerEilers:
            // Same: the tuner never routes penalized fits.
            smoother = "WhittakerEilers"
            reason = "Continuous responses; penalized fit selected explicitly."
        case .totalVariation:
            // Same: the tuner never routes edge-preserving fits.
            smoother = "TotalVariation"
            reason = "Continuous responses; edge-preserving fit selected explicitly."
        }
        return (fit, TuningSummary(smoother: smoother, detail: detail, score: score,
                                   reason: reason, notes: notes))
    }
}
