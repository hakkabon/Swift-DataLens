import Foundation

/// One equal-frequency bin in an out-of-fold binomial calibration display.
public struct CalibrationBin: Codable, Sendable, Hashable, Identifiable {
    public let id: Int
    public let observationCount: Int
    public let minimumPredictedProbability: Double
    public let maximumPredictedProbability: Double
    public let meanPredictedProbability: Double
    public let observedFrequency: Double

    public init(
        id: Int, observationCount: Int, minimumPredictedProbability: Double,
        maximumPredictedProbability: Double, meanPredictedProbability: Double,
        observedFrequency: Double
    ) {
        self.id = id
        self.observationCount = observationCount
        self.minimumPredictedProbability = minimumPredictedProbability
        self.maximumPredictedProbability = maximumPredictedProbability
        self.meanPredictedProbability = meanPredictedProbability
        self.observedFrequency = observedFrequency
    }
}

/// Calibration evidence computed exclusively from out-of-fold probabilities.
///
/// This is a descriptive reliability display: `expectedCalibrationError` is
/// binned and therefore changes with `binCount`. It is not a hypothesis test
/// or an in-sample goodness-of-fit claim.
public struct BinomialCalibration: Codable, Sendable, Hashable {
    public let observationCount: Int
    public let binCount: Int
    public let observedPrevalence: Double
    public let meanPredictedProbability: Double
    public let brierScore: Double
    public let expectedCalibrationError: Double
    public let bins: [CalibrationBin]
}

extension ModelValidation {
    /// Equal-frequency reliability bins for a binary model's held-out predictions.
    ///
    /// Returns `nil` for non-binomial validation, invalid bin counts, or
    /// predictions outside the probability range. Call this on the result of
    /// ``CrossValidation/evaluate(trainX:trainY:configuration:)`` rather than
    /// on training fitted values to avoid optimistic calibration displays.
    public func binomialCalibration(binCount: Int = 10) -> BinomialCalibration? {
        guard responseFamily == .binomial, binCount > 0, !predictions.isEmpty,
              predictions.allSatisfy({
                  ($0.observed == 0 || $0.observed == 1)
                      && $0.predicted.isFinite && $0.predicted >= 0 && $0.predicted <= 1
              }) else { return nil }
        let sorted = predictions.sorted {
            $0.predicted == $1.predicted ? $0.id < $1.id : $0.predicted < $1.predicted
        }
        let actualBinCount = min(binCount, sorted.count)
        let bins = (0..<actualBinCount).map { index -> CalibrationBin in
            let start = index * sorted.count / actualBinCount
            let end = (index + 1) * sorted.count / actualBinCount
            let values = Array(sorted[start..<end])
            let n = Double(values.count)
            let meanPrediction = values.reduce(0.0) { $0 + $1.predicted } / n
            let observed = values.reduce(0.0) { $0 + $1.observed } / n
            return CalibrationBin(
                id: index, observationCount: values.count,
                minimumPredictedProbability: values.map(\.predicted).min() ?? .nan,
                maximumPredictedProbability: values.map(\.predicted).max() ?? .nan,
                meanPredictedProbability: meanPrediction, observedFrequency: observed
            )
        }
        let n = Double(sorted.count)
        let prevalence = sorted.reduce(0.0) { $0 + $1.observed } / n
        let predictedMean = sorted.reduce(0.0) { $0 + $1.predicted } / n
        let brier = sorted.reduce(0.0) { $0 + pow($1.observed - $1.predicted, 2) } / n
        let ece = bins.reduce(0.0) {
            $0 + Double($1.observationCount) / n * abs($1.observedFrequency - $1.meanPredictedProbability)
        }
        return BinomialCalibration(
            observationCount: sorted.count, binCount: actualBinCount,
            observedPrevalence: prevalence, meanPredictedProbability: predictedMean,
            brierScore: brier, expectedCalibrationError: ece, bins: bins
        )
    }
}

/// Why two validation results cannot make a paired model-comparison claim.
public enum ValidationComparisonStatus: String, Codable, Sendable, Hashable {
    /// Family and every held-out observation/fold match exactly.
    case comparable
    /// Gaussian, binomial, and Poisson losses are not compared on one scale.
    case responseFamilyMismatch
    /// The validations do not contain precisely the same held-out rows/responses.
    case heldOutObservationsMismatch
    /// The held-out row mapping is the same but fold assignment differs.
    case foldAssignmentMismatch
}

/// Paired held-out loss for one observation in an honest model comparison.
public struct PairedValidationLoss: Codable, Sendable, Hashable, Identifiable {
    public let id: Int
    public let fold: Int
    public let baselineLoss: Double
    public let candidateLoss: Double
    /// Candidate minus baseline; negative favors the candidate.
    public let difference: Double
}

/// A same-task paired comparison suitable for a model-comparison panel.
///
/// It reports descriptive held-out loss differences, not a p-value. The
/// comparison is unavailable unless response family, held-out input IDs,
/// observed responses, and fold assignments all match exactly.
public struct ValidationComparison: Codable, Sendable, Hashable {
    public let status: ValidationComparisonStatus
    public let responseFamily: ResponseFamily?
    public let baselinePrimaryScore: Double?
    public let candidatePrimaryScore: Double?
    /// Mean candidate-minus-baseline pointwise loss; negative favors candidate.
    public let meanLossDifference: Double?
    public let candidateWinCount: Int
    public let baselineWinCount: Int
    public let tieCount: Int
    public let pairedLosses: [PairedValidationLoss]

    fileprivate init(
        status: ValidationComparisonStatus, responseFamily: ResponseFamily? = nil,
        baselinePrimaryScore: Double? = nil, candidatePrimaryScore: Double? = nil,
        meanLossDifference: Double? = nil, candidateWinCount: Int = 0,
        baselineWinCount: Int = 0, tieCount: Int = 0,
        pairedLosses: [PairedValidationLoss] = []
    ) {
        self.status = status
        self.responseFamily = responseFamily
        self.baselinePrimaryScore = baselinePrimaryScore
        self.candidatePrimaryScore = candidatePrimaryScore
        self.meanLossDifference = meanLossDifference
        self.candidateWinCount = candidateWinCount
        self.baselineWinCount = baselineWinCount
        self.tieCount = tieCount
        self.pairedLosses = pairedLosses
    }
}

/// Builds guarded paired comparisons from cross-validation evidence.
public enum ModelComparison {
    /// Compare two validations only when their out-of-fold tasks are identical.
    public static func compare(
        baseline: ModelValidation, candidate: ModelValidation
    ) -> ValidationComparison {
        guard baseline.responseFamily == candidate.responseFamily else {
            return .init(status: .responseFamilyMismatch)
        }
        let baselineRows = baseline.predictions.sorted { $0.id < $1.id }
        let candidateRows = candidate.predictions.sorted { $0.id < $1.id }
        guard baselineRows.count == candidateRows.count,
              zip(baselineRows, candidateRows).allSatisfy({
                  $0.id == $1.id && $0.observed == $1.observed
              }) else {
            return .init(status: .heldOutObservationsMismatch)
        }
        guard zip(baselineRows, candidateRows).allSatisfy({ $0.fold == $1.fold }) else {
            return .init(status: .foldAssignmentMismatch)
        }
        let family = baseline.responseFamily
        let pairs = zip(baselineRows, candidateRows).map { base, contender in
            let baselineLoss = loss(observed: base.observed, predicted: base.predicted, family: family)
            let candidateLoss = loss(observed: contender.observed, predicted: contender.predicted, family: family)
            return PairedValidationLoss(
                id: base.id, fold: base.fold, baselineLoss: baselineLoss,
                candidateLoss: candidateLoss, difference: candidateLoss - baselineLoss
            )
        }
        guard pairs.allSatisfy({ $0.baselineLoss.isFinite && $0.candidateLoss.isFinite }) else {
            return .init(status: .heldOutObservationsMismatch)
        }
        var candidateWins = 0
        var baselineWins = 0
        var ties = 0
        for pair in pairs {
            let tolerance = 1e-12 * max(1, abs(pair.baselineLoss), abs(pair.candidateLoss))
            if pair.difference < -tolerance {
                candidateWins += 1
            } else if pair.difference > tolerance {
                baselineWins += 1
            } else {
                ties += 1
            }
        }
        return ValidationComparison(
            status: .comparable, responseFamily: family,
            baselinePrimaryScore: baseline.primaryScore, candidatePrimaryScore: candidate.primaryScore,
            meanLossDifference: pairs.reduce(0) { $0 + $1.difference } / Double(pairs.count),
            candidateWinCount: candidateWins, baselineWinCount: baselineWins,
            tieCount: ties, pairedLosses: pairs
        )
    }

    private static func loss(observed: Double, predicted: Double, family: ResponseFamily) -> Double {
        switch family {
        case .gaussian:
            return pow(observed - predicted, 2)
        case .binomial:
            let mean = min(max(predicted, 1e-15), 1 - 1e-15)
            return -2 * (observed * log(mean) + (1 - observed) * log(1 - mean))
        case .poisson:
            let mean = max(predicted, 1e-15)
            return observed == 0 ? 2 * mean : 2 * (observed * log(observed / mean) - (observed - mean))
        }
    }
}
