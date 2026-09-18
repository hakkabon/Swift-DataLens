import Foundation

/// Statistical response family used by a fitted smoother.
public enum ResponseFamily: String, Codable, Hashable, Sendable {
    case gaussian
    case binomial
    case poisson
}

/// Link function mapping the response mean to the linear predictor.
public enum LinkFunction: String, Codable, Hashable, Sendable {
    case identity
    case logit
    case log
}

/// Residual definition available for model checking.
public enum ResidualKind: String, Codable, Hashable, Sendable {
    /// Observed response minus fitted response mean.
    case raw
    /// Raw residual divided by the model variance scale.
    case pearson
    /// Signed square root of the observation's unit-deviance contribution.
    case deviance
}

/// Stable, serializable statistical metadata shared by every smoother.
public struct FitDiagnostics: Codable, Hashable, Sendable {
    public let responseFamily: ResponseFamily
    public let linkFunction: LinkFunction
    public let observationCount: Int
    public let effectiveDegreesOfFreedom: Double
    public let residualScale: Double
    public let deviance: Double?
    public let nullDeviance: Double?
    public let availableResiduals: [ResidualKind]

    public init(responseFamily: ResponseFamily, linkFunction: LinkFunction,
                observationCount: Int, effectiveDegreesOfFreedom: Double,
                residualScale: Double, deviance: Double? = nil,
                nullDeviance: Double? = nil,
                availableResiduals: [ResidualKind] = [.raw, .pearson, .deviance]) {
        self.responseFamily = responseFamily
        self.linkFunction = linkFunction
        self.observationCount = observationCount
        self.effectiveDegreesOfFreedom = effectiveDegreesOfFreedom
        self.residualScale = residualScale
        self.deviance = deviance
        self.nullDeviance = nullDeviance
        self.availableResiduals = availableResiduals
    }
}

extension FittedSmoother {
    /// Model-family, link, scale, and goodness-of-fit metadata suitable for
    /// diagnostics, provenance records, and exported analysis reports.
    public var diagnostics: FitDiagnostics {
        let common = commonDiagnosticValues
        if case .likelihood(let fit) = self {
            switch fit.family {
            case .gaussian:
                return FitDiagnostics(responseFamily: .gaussian, linkFunction: .identity,
                                      observationCount: common.y.count,
                                      effectiveDegreesOfFreedom: common.trace,
                                      residualScale: common.sigma,
                                      deviance: fit.deviance, nullDeviance: fit.nullDeviance)
            case .binomial:
                return FitDiagnostics(responseFamily: .binomial, linkFunction: .logit,
                                      observationCount: common.y.count,
                                      effectiveDegreesOfFreedom: common.trace,
                                      residualScale: 1, deviance: fit.deviance,
                                      nullDeviance: fit.nullDeviance)
            case .poisson:
                return FitDiagnostics(responseFamily: .poisson, linkFunction: .log,
                                      observationCount: common.y.count,
                                      effectiveDegreesOfFreedom: common.trace,
                                      residualScale: 1, deviance: fit.deviance,
                                      nullDeviance: fit.nullDeviance)
            }
        }
        return FitDiagnostics(responseFamily: .gaussian, linkFunction: .identity,
                              observationCount: common.y.count,
                              effectiveDegreesOfFreedom: common.trace,
                              residualScale: common.sigma)
    }

    /// Residuals at the retained training observations under a documented,
    /// family-appropriate definition. Values align with ``keptIndices``.
    public func residuals(_ kind: ResidualKind) -> [Double] {
        let common = commonDiagnosticValues
        let family = diagnostics.responseFamily
        let epsilon = 1e-15
        return zip(common.y, common.fitted).map { y, fitted in
            let raw = y - fitted
            switch (kind, family) {
            case (.raw, _):
                return raw
            case (.pearson, .gaussian):
                return common.sigma > 0 ? raw / common.sigma : (raw == 0 ? 0 : .nan)
            case (.pearson, .binomial):
                let mu = min(max(fitted, epsilon), 1 - epsilon)
                return raw / sqrt(mu * (1 - mu))
            case (.pearson, .poisson):
                return raw / sqrt(max(fitted, epsilon))
            case (.deviance, .gaussian):
                return raw
            case (.deviance, .binomial):
                let mu = min(max(fitted, epsilon), 1 - epsilon)
                let first = y == 0 ? 0 : y * log(y / mu)
                let second = y == 1 ? 0 : (1 - y) * log((1 - y) / (1 - mu))
                return (raw < 0 ? -1 : raw > 0 ? 1 : 0) * sqrt(max(2 * (first + second), 0))
            case (.deviance, .poisson):
                let mu = max(fitted, epsilon)
                let contribution = y == 0 ? mu : y * log(y / mu) - (y - mu)
                return (raw < 0 ? -1 : raw > 0 ? 1 : 0) * sqrt(max(2 * contribution, 0))
            }
        }
    }

    private var commonDiagnosticValues: (y: [Double], fitted: [Double], sigma: Double, trace: Double) {
        switch self {
        case .loess(let fit): (fit.trainY, fit.fittedValues, fit.sigma, fit.trace)
        case .adaptive(let fit): (fit.trainY, fit.fittedValues, fit.sigma, fit.trace)
        case .likelihood(let fit): (fit.trainY, fit.fittedValues, fit.sigma, fit.trace)
        case .nadarayaWatson(let fit): (fit.trainY, fit.fittedValues, fit.sigma, fit.trace)
        case .whittakerEilers(let fit): (fit.trainY, fit.fittedValues, fit.sigma, fit.trace)
        case .totalVariation(let fit): (fit.trainY, fit.fittedValues, fit.sigma, fit.trace)
        }
    }
}
