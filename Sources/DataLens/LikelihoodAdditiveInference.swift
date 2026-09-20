import Foundation

/// The covariance approximation used by a likelihood additive model.
///
/// This is conditional on the fitted spline basis and fixed penalty weight.
/// It does not include smoothing-selection, model-selection, or resampling
/// uncertainty; use ``ModelResampling/bootstrap(trainX:trainY:queryPoints:configuration:)``
/// when that distinction matters.
public enum LikelihoodAdditiveCovarianceMethod: String, Codable, Sendable, Hashable {
    /// Sandwich covariance from the final penalized IRLS observed information.
    case penalizedObservedInformation
}

/// Conditional covariance and effective degrees of freedom for a likelihood GAM.
///
/// Coefficients are ordered as the intercept followed by each term's spline
/// coefficients in ``LikelihoodAdditiveModel/terms`` order. The covariance is
/// a fixed-basis, fixed-penalty approximation for the canonical binomial or
/// Poisson family, not a post-selection confidence statement.
public struct LikelihoodAdditiveInference: Sendable {
    public let covarianceMethod: LikelihoodAdditiveCovarianceMethod
    /// Trace of the final penalized IRLS hat matrix.
    public let effectiveDegreesOfFreedom: Double
    /// Approximate covariance matrix in documented coefficient order.
    public let coefficientCovariance: [[Double]]
    /// Square roots of the covariance diagonal in documented coefficient order.
    public let coefficientStandardErrors: [Double]

    init(
        covarianceMethod: LikelihoodAdditiveCovarianceMethod,
        effectiveDegreesOfFreedom: Double, coefficientCovariance: [[Double]]
    ) {
        self.covarianceMethod = covarianceMethod
        self.effectiveDegreesOfFreedom = effectiveDegreesOfFreedom
        self.coefficientCovariance = coefficientCovariance
        coefficientStandardErrors = coefficientCovariance.indices.map {
            sqrt(max(coefficientCovariance[$0][$0], 0))
        }
    }
}

/// A two-sided interval with its point estimate and nominal confidence level.
///
/// The producing API documents whether this is a conditional normal-approximation
/// interval or a bootstrap percentile interval.
public struct StatisticalInterval: Codable, Sendable, Hashable {
    public let estimate: Double
    public let lowerBound: Double
    public let upperBound: Double
    public let confidenceLevel: Double

    public init(estimate: Double, lowerBound: Double, upperBound: Double, confidenceLevel: Double) {
        self.estimate = estimate
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.confidenceLevel = confidenceLevel
    }
}

/// A conditional link-scale interval for one centered likelihood-GAM term.
public struct LikelihoodAdditivePartialEffectInterval: Codable, Sendable, Hashable {
    public let predictorIndex: Int
    public let x: [Double]
    public let effect: [Double]
    public let lowerBounds: [Double]
    public let upperBounds: [Double]
    public let confidenceLevel: Double
}

enum InferenceMath {
    static func quadraticForm(_ vector: [Double], covariance: [[Double]]) -> Double? {
        guard covariance.count == vector.count,
              covariance.allSatisfy({ $0.count == vector.count }) else { return nil }
        var result = 0.0
        for row in vector.indices {
            for column in vector.indices {
                result += vector[row] * covariance[row][column] * vector[column]
            }
        }
        return result.isFinite ? max(result, 0) : nil
    }

    /// Acklam's rational approximation to Φ⁻¹(p), accurate enough for
    /// displayed normal-approximation intervals without adding a platform API.
    static func normalQuantile(_ probability: Double) -> Double? {
        guard probability > 0, probability < 1, probability.isFinite else { return nil }
        let a = [-3.969_683_028_665_376e+01, 2.209_460_984_245_205e+02,
                 -2.759_285_104_469_687e+02, 1.383_577_518_672_690e+02,
                 -3.066_479_806_614_716e+01, 2.506_628_277_459_239]
        let b = [-5.447_609_879_822_406e+01, 1.615_858_368_580_409e+02,
                 -1.556_989_798_598_866e+02, 6.680_131_188_771_972e+01,
                 -1.328_068_155_288_572e+01]
        let c = [-7.784_894_002_430_293e-03, -3.223_964_580_411_365e-01,
                 -2.400_758_277_161_838, -2.549_732_539_343_734,
                 4.374_664_141_464_968, 2.938_163_982_698_783]
        let d = [7.784_695_709_041_462e-03, 3.224_671_290_700_398e-01,
                 2.445_134_137_142_996, 3.754_408_661_907_416]
        let lower = 0.02425
        let upper = 1 - lower
        if probability < lower {
            let q = sqrt(-2 * log(probability))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        }
        if probability > upper {
            let q = sqrt(-2 * log(1 - probability))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        }
        let q = probability - 0.5
        let r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    }

    static func percentile(_ values: [Double], probability: Double) -> Double? {
        guard !values.isEmpty, probability >= 0, probability <= 1, probability.isFinite,
              values.allSatisfy(\.isFinite) else { return nil }
        let sorted = values.sorted()
        let position = probability * Double(sorted.count - 1)
        let low = Int(position.rounded(.down))
        let high = Int(position.rounded(.up))
        let fraction = position - Double(low)
        return sorted[low] + fraction * (sorted[high] - sorted[low])
    }
}
