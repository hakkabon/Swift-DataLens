import Foundation

/// Reproducible nonparametric-bootstrap settings for one unified model specification.
public struct BootstrapConfiguration: Codable, Sendable, Hashable {
    public let replicateCount: Int
    public let minimumSuccessFraction: Double
    public let confidenceLevel: Double
    public let seed: UInt64
    public let specification: StatisticalModelSpecification

    public init(
        replicateCount: Int = 200, minimumSuccessFraction: Double = 0.8,
        confidenceLevel: Double = 0.95, seed: UInt64 = 0xB005_7A9,
        specification: StatisticalModelSpecification = StatisticalModelSpecification()
    ) {
        self.replicateCount = replicateCount
        self.minimumSuccessFraction = minimumSuccessFraction
        self.confidenceLevel = confidenceLevel
        self.seed = seed
        self.specification = specification
    }

    var isValid: Bool {
        replicateCount >= 2 && minimumSuccessFraction.isFinite
            && minimumSuccessFraction > 0 && minimumSuccessFraction <= 1
            && confidenceLevel > 0 && confidenceLevel < 1 && confidenceLevel.isFinite
            && specification.isValid
    }
}

/// Explicit stop verdict for a bootstrap stability run.
public enum BootstrapStatus: String, Codable, Sendable, Hashable {
    case completed
    case invalidInput
    case referenceFitFailure
    case insufficientSuccessfulReplicates
}

/// Bootstrap stability evidence for one requested prediction point.
///
/// `interval` is a percentile interval over successful replica fits. It does
/// not correct for bias or acceleration and should be described as such in a
/// report.
public struct BootstrapPrediction: Codable, Sendable, Hashable, Identifiable {
    public let id: Int
    public let query: [Double]
    public let referenceEstimate: Double
    public let bootstrapMean: Double
    public let bootstrapStandardDeviation: Double
    public let interval: StatisticalInterval
}

/// Result of deterministic bootstrap refitting with its fit-failure accounting.
///
/// A result exposes intervals only when enough replicas meet the configured
/// success threshold. The attempted and failed counts are retained so a panel
/// cannot present a sparse successful subset as a complete bootstrap.
public struct BootstrapResult: Codable, Sendable, Hashable {
    public let status: BootstrapStatus
    public let responseFamily: ResponseFamily?
    public let configuration: BootstrapConfiguration
    public let attemptedReplicates: Int
    public let successfulReplicates: Int
    public let failedReplicates: Int
    public let predictions: [BootstrapPrediction]

    /// Fraction of requested replicas that produced finite predictions.
    public var successFraction: Double {
        attemptedReplicates > 0 ? Double(successfulReplicates) / Double(attemptedReplicates) : 0
    }

    fileprivate init(
        status: BootstrapStatus, responseFamily: ResponseFamily?, configuration: BootstrapConfiguration,
        attemptedReplicates: Int, successfulReplicates: Int, predictions: [BootstrapPrediction] = []
    ) {
        self.status = status
        self.responseFamily = responseFamily
        self.configuration = configuration
        self.attemptedReplicates = attemptedReplicates
        self.successfulReplicates = successfulReplicates
        failedReplicates = attemptedReplicates - successfulReplicates
        self.predictions = predictions
    }
}

/// Deterministic bootstrap refits for prediction stability and percentile intervals.
public enum ModelResampling {
    /// Bootstrap a complete configured workflow at fixed query points.
    ///
    /// The reference fit first drops invalid rows through the normal unified
    /// fit path. Replicas sample only those retained rows with replacement,
    /// then replay the exact saved specification (including tuning where it
    /// is part of that specification). Failed or non-finite replica fits are
    /// counted, never silently discarded. This operation is intentionally
    /// sequential to make a supplied seed reproducible bit-for-bit.
    public static func bootstrap(
        trainX: [[Double]], trainY: [Double], queryPoints: [[Double]],
        configuration: BootstrapConfiguration = BootstrapConfiguration()
    ) -> BootstrapResult {
        guard configuration.isValid, !queryPoints.isEmpty else {
            return BootstrapResult(
                status: .invalidInput, responseFamily: nil, configuration: configuration,
                attemptedReplicates: 0, successfulReplicates: 0
            )
        }
        guard let reference = FittedStatisticalModel.fit(
            trainX: trainX, trainY: trainY, specification: configuration.specification
        ) else {
            return BootstrapResult(
                status: .referenceFitFailure, responseFamily: nil, configuration: configuration,
                attemptedReplicates: 0, successfulReplicates: 0
            )
        }
        guard let width = reference.trainingPredictors.first?.count,
              queryPoints.allSatisfy({ $0.count == width && $0.allSatisfy(\.isFinite) }) else {
            return BootstrapResult(
                status: .invalidInput, responseFamily: reference.diagnostics.responseFamily,
                configuration: configuration, attemptedReplicates: 0, successfulReplicates: 0
            )
        }
        let referencePredictions = reference.predict(queryPoints)
        guard referencePredictions.allSatisfy(\.isFinite) else {
            return BootstrapResult(
                status: .referenceFitFailure, responseFamily: reference.diagnostics.responseFamily,
                configuration: configuration, attemptedReplicates: 0, successfulReplicates: 0
            )
        }
        let retainedX = reference.trainingPredictors
        let retainedY = reference.trainingResponses
        guard retainedX.count >= 2 else {
            return BootstrapResult(
                status: .referenceFitFailure, responseFamily: reference.diagnostics.responseFamily,
                configuration: configuration, attemptedReplicates: 0, successfulReplicates: 0
            )
        }
        var rng = SeedableRandomNumberGenerator(seed: configuration.seed)
        var replicaPredictions = Array(repeating: [Double](), count: queryPoints.count)
        var successes = 0
        for _ in 0..<configuration.replicateCount {
            let draw = (0..<retainedX.count).map { _ in rng.nextInt(in: 0..<retainedX.count) }
            let sampleX = draw.map { retainedX[$0] }
            let sampleY = draw.map { retainedY[$0] }
            guard let replica = FittedStatisticalModel.fit(
                trainX: sampleX, trainY: sampleY, specification: configuration.specification
            ) else { continue }
            let prediction = replica.predict(queryPoints)
            guard prediction.count == queryPoints.count, prediction.allSatisfy(\.isFinite) else { continue }
            for index in prediction.indices { replicaPredictions[index].append(prediction[index]) }
            successes += 1
        }
        let requiredSuccesses = Int(ceil(configuration.minimumSuccessFraction * Double(configuration.replicateCount)))
        guard successes >= requiredSuccesses else {
            return BootstrapResult(
                status: .insufficientSuccessfulReplicates,
                responseFamily: reference.diagnostics.responseFamily, configuration: configuration,
                attemptedReplicates: configuration.replicateCount, successfulReplicates: successes
            )
        }
        let lowerProbability = (1 - configuration.confidenceLevel) / 2
        let upperProbability = 1 - lowerProbability
        let summaries = queryPoints.indices.compactMap { index -> BootstrapPrediction? in
            let values = replicaPredictions[index]
            guard let lower = InferenceMath.percentile(values, probability: lowerProbability),
                  let upper = InferenceMath.percentile(values, probability: upperProbability) else { return nil }
            let n = Double(values.count)
            let mean = values.reduce(0, +) / n
            let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / Double(max(values.count - 1, 1))
            return BootstrapPrediction(
                id: index, query: queryPoints[index], referenceEstimate: referencePredictions[index],
                bootstrapMean: mean, bootstrapStandardDeviation: sqrt(max(variance, 0)),
                interval: StatisticalInterval(
                    estimate: referencePredictions[index], lowerBound: lower, upperBound: upper,
                    confidenceLevel: configuration.confidenceLevel
                )
            )
        }
        guard summaries.count == queryPoints.count else {
            return BootstrapResult(
                status: .insufficientSuccessfulReplicates,
                responseFamily: reference.diagnostics.responseFamily, configuration: configuration,
                attemptedReplicates: configuration.replicateCount, successfulReplicates: successes
            )
        }
        return BootstrapResult(
            status: .completed, responseFamily: reference.diagnostics.responseFamily,
            configuration: configuration, attemptedReplicates: configuration.replicateCount,
            successfulReplicates: successes, predictions: summaries
        )
    }
}
