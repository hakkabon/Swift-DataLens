import Foundation

/// How held-out rows are assigned to validation folds.
public enum ValidationPartitioning: String, Codable, Sendable, Hashable {
    /// Deterministic random assignment, appropriate for exchangeable rows.
    case shuffled
    /// Contiguous source-order blocks, appropriate for ordered or temporal rows.
    case blocked
    /// Deterministic, class-balanced assignment for binary responses.
    case stratifiedBinary
}

/// Reproducible cross-validation settings for a statistical-model specification.
public struct ValidationConfiguration: Codable, Sendable, Hashable {
    public let foldCount: Int
    public let partitioning: ValidationPartitioning
    public let seed: UInt64
    public let specification: StatisticalModelSpecification

    public init(
        foldCount: Int = 5, partitioning: ValidationPartitioning = .shuffled,
        seed: UInt64 = 0xDADA_2026, specification: StatisticalModelSpecification = StatisticalModelSpecification()
    ) {
        self.foldCount = foldCount
        self.partitioning = partitioning
        self.seed = seed
        self.specification = specification
    }
}

/// One held-out prediction, including its original input-row position.
public struct OutOfFoldPrediction: Codable, Sendable, Hashable, Identifiable {
    public let id: Int
    public let fold: Int
    public let observed: Double
    public let predicted: Double

    public init(inputIndex: Int, fold: Int, observed: Double, predicted: Double) {
        id = inputIndex
        self.fold = fold
        self.observed = observed
        self.predicted = predicted
    }
}

/// Metrics computed on one held-out fold.
public struct ValidationFold: Codable, Sendable, Hashable, Identifiable {
    public let id: Int
    public let observationCount: Int
    public let rootMeanSquaredError: Double?
    public let meanAbsoluteError: Double?
    public let meanDeviance: Double?

    public init(
        id: Int, observationCount: Int, rootMeanSquaredError: Double?,
        meanAbsoluteError: Double?, meanDeviance: Double?
    ) {
        self.id = id
        self.observationCount = observationCount
        self.rootMeanSquaredError = rootMeanSquaredError
        self.meanAbsoluteError = meanAbsoluteError
        self.meanDeviance = meanDeviance
    }
}

/// Out-of-fold evidence for one statistical-model specification.
///
/// Gaussian fits report RMSE and MAE. Binomial and Poisson fits report mean
/// unit deviance, avoiding an R²-style score whose interpretation does not
/// carry across response families.
public struct ModelValidation: Codable, Sendable, Hashable {
    public let responseFamily: ResponseFamily
    public let configuration: ValidationConfiguration
    public let retainedObservationCount: Int
    public let folds: [ValidationFold]
    public let predictions: [OutOfFoldPrediction]
    public let rootMeanSquaredError: Double?
    public let meanAbsoluteError: Double?
    public let meanDeviance: Double?

    /// The primary family-appropriate score: RMSE for Gaussian, mean deviance
    /// for binomial/Poisson. Lower is better in every case.
    public var primaryScore: Double? {
        responseFamily == .gaussian ? rootMeanSquaredError : meanDeviance
    }
}

/// Deterministic out-of-fold evaluation of ``StatisticalModelSpecification``.
///
/// This validates the entire configured fit. For automatic smoothing that
/// includes response routing and tuning; for an additive specification it
/// includes term selection and cyclic backfitting. In either case held-out
/// rows never influence the model fitted for their prediction.
public enum CrossValidation {
    /// Evaluate a statistical model with held-out predictions.
    ///
    /// Returns `nil` for invalid dimensions, insufficient data, non-finite
    /// rows, an incompatible partitioning, or any fold that cannot produce a
    /// finite fit. It fails closed rather than quietly dropping a bad fold.
    public static func evaluate(
        trainX: [[Double]], trainY: [Double],
        configuration: ValidationConfiguration = ValidationConfiguration()
    ) -> ModelValidation? {
        guard trainX.count == trainY.count, trainX.count >= 3,
              configuration.foldCount >= 2, configuration.foldCount <= trainX.count,
              configuration.specification.isValid else { return nil }
        let finiteRows = trainX.indices.compactMap { index -> (Int, [Double], Double)? in
            let row = trainX[index]
            guard !row.isEmpty, row.allSatisfy(\.isFinite), trainY[index].isFinite else { return nil }
            return (index, row, trainY[index])
        }
        guard finiteRows.count >= configuration.foldCount,
              let width = finiteRows.first?.1.count,
              finiteRows.allSatisfy({ $0.1.count == width }) else { return nil }
        // A requested GAM is a Gaussian identity model by contract, even when
        // its response values are integral. Automatic smoothing retains its
        // data-driven binary/count routing.
        let family: ResponseFamily = configuration.specification.strategy == .additiveGaussian
            ? .gaussian : responseFamily(for: finiteRows.map(\.2))
        guard configuration.partitioning != .stratifiedBinary || family == .binomial else { return nil }
        let heldOutFolds = makeFolds(
            rows: finiteRows, foldCount: configuration.foldCount,
            partitioning: configuration.partitioning, seed: configuration.seed
        )
        guard heldOutFolds.count == configuration.foldCount,
              heldOutFolds.allSatisfy({ !$0.isEmpty }) else { return nil }

        var predictions: [OutOfFoldPrediction] = []
        var foldMetrics: [ValidationFold] = []
        predictions.reserveCapacity(finiteRows.count)
        foldMetrics.reserveCapacity(heldOutFolds.count)
        for foldIndex in heldOutFolds.indices {
            let heldOut = heldOutFolds[foldIndex]
            let heldOutIDs = Set(heldOut.map(\.0))
            let training = finiteRows.filter { !heldOutIDs.contains($0.0) }
            guard !training.isEmpty,
                  let model = FittedStatisticalModel.fit(
                    trainX: training.map(\.1), trainY: training.map(\.2),
                    specification: configuration.specification
                  ) else { return nil }
            let predicted = model.predict(heldOut.map(\.1), extrapolation: .polynomial)
            guard predicted.count == heldOut.count, predicted.allSatisfy(\.isFinite) else { return nil }
            let foldPredictions = zip(heldOut, predicted).map {
                OutOfFoldPrediction(inputIndex: $0.0.0, fold: foldIndex,
                                     observed: $0.0.2, predicted: $0.1)
            }
            predictions += foldPredictions
            foldMetrics.append(metrics(for: foldPredictions, family: family, id: foldIndex))
        }
        predictions.sort { $0.id < $1.id }
        let allMetrics = metrics(for: predictions, family: family, id: 0)
        return ModelValidation(
            responseFamily: family, configuration: configuration,
            retainedObservationCount: finiteRows.count, folds: foldMetrics,
            predictions: predictions, rootMeanSquaredError: allMetrics.rootMeanSquaredError,
            meanAbsoluteError: allMetrics.meanAbsoluteError, meanDeviance: allMetrics.meanDeviance
        )
    }

    private static func responseFamily(for values: [Double]) -> ResponseFamily {
        if values.allSatisfy({ $0 == 0 || $0 == 1 }) { return .binomial }
        if values.allSatisfy({ $0 >= 0 && $0 == $0.rounded() }) { return .poisson }
        return .gaussian
    }

    private static func makeFolds(
        rows: [(Int, [Double], Double)], foldCount: Int,
        partitioning: ValidationPartitioning, seed: UInt64
    ) -> [[(Int, [Double], Double)]] {
        var folds = Array(repeating: [(Int, [Double], Double)](), count: foldCount)
        switch partitioning {
        case .blocked:
            for (index, row) in rows.enumerated() {
                folds[min(index * foldCount / rows.count, foldCount - 1)].append(row)
            }
        case .shuffled:
            var shuffled = rows
            shuffle(&shuffled, seed: seed)
            for (index, row) in shuffled.enumerated() { folds[index % foldCount].append(row) }
        case .stratifiedBinary:
            var zeroes = rows.filter { $0.2 == 0 }
            var ones = rows.filter { $0.2 == 1 }
            shuffle(&zeroes, seed: seed)
            shuffle(&ones, seed: seed ^ 0x9E37_79B9_7F4A_7C15)
            for (index, row) in zeroes.enumerated() { folds[index % foldCount].append(row) }
            for (index, row) in ones.enumerated() { folds[index % foldCount].append(row) }
        }
        return folds
    }

    private static func shuffle<T>(_ values: inout [T], seed: UInt64) {
        guard values.count > 1 else { return }
        var state = seed == 0 ? 0xA076_1D64_78BD_642F : seed
        for index in values.indices.dropFirst().reversed() {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let target = Int(state % UInt64(index + 1))
            values.swapAt(index, target)
        }
    }

    private static func metrics(
        for predictions: [OutOfFoldPrediction], family: ResponseFamily, id: Int
    ) -> ValidationFold {
        let n = Double(predictions.count)
        switch family {
        case .gaussian:
            let residuals = predictions.map { $0.observed - $0.predicted }
            let rmse = sqrt(residuals.reduce(0) { $0 + $1 * $1 } / n)
            let mae = residuals.reduce(0) { $0 + abs($1) } / n
            return ValidationFold(id: id, observationCount: predictions.count,
                                  rootMeanSquaredError: rmse, meanAbsoluteError: mae,
                                  meanDeviance: nil)
        case .binomial:
            let epsilon = 1e-15
            let deviance = predictions.reduce(0.0) { sum, prediction in
                let mean = min(max(prediction.predicted, epsilon), 1 - epsilon)
                let y = prediction.observed
                return sum - 2 * (y * log(mean) + (1 - y) * log(1 - mean))
            }
            return ValidationFold(id: id, observationCount: predictions.count,
                                  rootMeanSquaredError: nil, meanAbsoluteError: nil,
                                  meanDeviance: deviance / n)
        case .poisson:
            let epsilon = 1e-15
            let deviance = predictions.reduce(0.0) { sum, prediction in
                let mean = max(prediction.predicted, epsilon)
                let y = prediction.observed
                return sum + (y == 0 ? 2 * mean : 2 * (y * log(y / mean) - (y - mean)))
            }
            return ValidationFold(id: id, observationCount: predictions.count,
                                  rootMeanSquaredError: nil, meanAbsoluteError: nil,
                                  meanDeviance: deviance / n)
        }
    }
}
