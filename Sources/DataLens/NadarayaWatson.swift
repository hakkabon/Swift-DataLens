import Foundation

/// Nadaraya–Watson kernel regression: locally constant fits under tricube
/// nearest-neighbor weights with bisquare robustness iterations.
///
/// Deliberately thin: every local evaluation delegates to
/// `Loess.localFit` with degree 0 (identical values, fallbacks, and
/// leverage), and standard errors go through
/// `Loess.kernelStandardError` the same way. What this type owns is the
/// estimator's contract — span-fraction neighborhoods (so the tuner can
/// compare it against `Loess` directly), the smoother trace, and analytic
/// gradients of the kernel mean — plus the usual single/batch/concurrent
/// evaluation surface.
///
/// Span semantics mirror `Loess`: the neighborhood holds
/// `ceil(span · n)` points (at least 2). Like `Loess`, the bandwidth is
/// the max neighbor distance, so flat regions automatically go wide.
public struct NadarayaWatson: Sendable {
    public let trainX: [[Double]]
    public let trainY: [Double]
    public let span: Double
    /// Fitted values at the training points (final robustness iteration).
    public let fittedValues: [Double]
    /// Residual scale σ̂ = √(RSS/max(n−tr,1)).
    public let sigma: Double
    /// Smoother-matrix trace (effective degrees of freedom).
    public let trace: Double
    /// Final robustness weights (1 where nothing was downweighted).
    public let weights: [Double]
    /// Survivor positions for `droppingMissing` (identity otherwise).
    public let keptIndices: [Int]

    init(trainX: [[Double]], trainY: [Double], span: Double,
         fittedValues: [Double], sigma: Double, trace: Double,
         weights: [Double], keptIndices: [Int]) {
        self.trainX = trainX
        self.trainY = trainY
        self.span = span
        self.fittedValues = fittedValues
        self.sigma = sigma
        self.trace = trace
        self.weights = weights
        self.keptIndices = keptIndices
    }

    /// Neighborhood size for `n` rows at `span` (at least 2 points: a
    /// constant needs a comparison to be a fit).
    static func neighborhoodCount(n: Int, span: Double) -> Int {
        min(n, max(Int(ceil(span * Double(n))), 2))
    }

    /// Fit by `robustIterations` bisquare reweighting rounds (4 matches R).
    ///
    /// With `droppingMissing`, rows with non-finite coordinates or responses
    /// are dropped first (`keptIndices` records the survivors); otherwise
    /// such rows fail validation and the fit is nil.
    public static func fit(trainX: [[Double]], trainY: [Double],
                           span: Double = 0.75,
                           robustIterations: Int = 4,
                           droppingMissing: Bool = false) -> NadarayaWatson? {
        guard trainX.count == trainY.count else { return nil }
        let (trainX, trainY, keptIndices): ([[Double]], [Double], [Int]) = droppingMissing
            ? MissingData.dropping(trainX: trainX, trainY: trainY)
            : (trainX, trainY, Array(trainX.indices))
        guard !trainX.isEmpty,
              span > 0, span <= 1,
              trainX.allSatisfy({ $0.count == trainX[0].count }),
              trainX.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              trainY.allSatisfy({ $0.isFinite }) else { return nil }
        let n = trainX.count
        let k = neighborhoodCount(n: n, span: span)
        guard k > 1 else { return nil }
        // One neighbor index per fit (not per query): the tree build is
        // O(n log n), so rebuilding it per local fit would lose to brute force.
        let search = NeighborSearch(trainX: trainX)
        var robust = [Double](repeating: 1, count: n)
        var fitted = [Double](repeating: 0, count: n)
        // Scale floor: once the fit is (near-)exact, MAD → 0 and an unguarded
        // cutoff would downweight good points. Floor against the y-scale.
        let medY = Descriptive.median(trainY) ?? 0
        let yScale = max(Descriptive.median(trainY.map { abs($0 - medY) }) ?? 0, 1e-300)
        for _ in 0...robustIterations {
            for i in 0..<n {
                fitted[i] = Loess.localFit(search: search, trainY: trainY, degree: 0,
                                           at: trainX[i], neighborhood: k, robust: robust).value
            }
            let resid = zip(trainY, fitted).map { abs($0 - $1) }
            guard let s = Descriptive.median(resid) else { break }
            let sEff = max(s, 1e-8 * yScale)
            robust = resid.map { LoessWeight.bisquare($0 / (6 * sEff)) }
        }
        // Final pass with diagnostics (weights[i] = own locality (=1) × robustness).
        var trace = 0.0
        for i in 0..<n {
            let r = Loess.localFit(search: search, trainY: trainY, degree: 0,
                                   at: trainX[i], neighborhood: k, robust: robust, trackIndex: i)
            fitted[i] = r.value
            trace += r.leverage
        }
        let rss = zip(trainY, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma = sqrt(rss / max(Double(n) - trace, 1))
        return NadarayaWatson(trainX: trainX, trainY: trainY, span: span,
                              fittedValues: fitted, sigma: sigma, trace: trace, weights: robust,
                              keptIndices: keptIndices)
    }

    /// Concurrent fit: identical to `fit(trainX:trainY:span:robustIterations:)`.
    ///
    /// Robustness rounds stay sequential (each round needs the previous
    /// residuals), but the per-point local fits within a round run in
    /// parallel. Bit-identical to `fit`.
    public static func fitConcurrently(trainX: [[Double]], trainY: [Double],
                                       span: Double = 0.75,
                                       robustIterations: Int = 4,
                                       droppingMissing: Bool = false) async throws -> NadarayaWatson? {
        guard trainX.count == trainY.count else { return nil }
        let (trainX, trainY, keptIndices): ([[Double]], [Double], [Int]) = droppingMissing
            ? MissingData.dropping(trainX: trainX, trainY: trainY)
            : (trainX, trainY, Array(trainX.indices))
        guard !trainX.isEmpty,
              span > 0, span <= 1,
              trainX.allSatisfy({ $0.count == trainX[0].count }),
              trainX.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              trainY.allSatisfy({ $0.isFinite }) else { return nil }
        let n = trainX.count
        let k = neighborhoodCount(n: n, span: span)
        guard k > 1 else { return nil }
        let search = NeighborSearch(trainX: trainX)
        var robust = [Double](repeating: 1, count: n)
        var fitted = [Double](repeating: 0, count: n)
        let medY = Descriptive.median(trainY) ?? 0
        let yScale = max(Descriptive.median(trainY.map { abs($0 - medY) }) ?? 0, 1e-300)
        for _ in 0...robustIterations {
            let currentRobust = robust
            let vals: [Double] = try await concurrentMap(over: n) { i in
                Loess.localFit(search: search, trainY: trainY, degree: 0,
                               at: trainX[i], neighborhood: k, robust: currentRobust).value
            }
            fitted = vals
            let resid = zip(trainY, fitted).map { abs($0 - $1) }
            guard let s = Descriptive.median(resid) else { break }
            let sEff = max(s, 1e-8 * yScale)
            robust = resid.map { LoessWeight.bisquare($0 / (6 * sEff)) }
        }
        let currentRobust = robust
        let final: [(value: Double, leverage: Double)] = try await concurrentMap(over: n) { i in
            Loess.localFit(search: search, trainY: trainY, degree: 0,
                           at: trainX[i], neighborhood: k, robust: currentRobust, trackIndex: i)
        }
        var trace = 0.0
        for (i, r) in final.enumerated() {
            fitted[i] = r.value
            trace += r.leverage
        }
        let rss = zip(trainY, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma = sqrt(rss / max(Double(n) - trace, 1))
        return NadarayaWatson(trainX: trainX, trainY: trainY, span: span,
                              fittedValues: fitted, sigma: sigma, trace: trace, weights: robust,
                              keptIndices: keptIndices)
    }

    /// Predict at `x` using the final robust weights (fallback cascade inside).
    ///
    /// `extrapolation` governs outside-hull queries (default `.polynomial`
    /// evaluates the kernel anyway — the local constant extends flatly).
    public func predict(_ x: [Double], extrapolation: ExtrapolationPolicy = .polynomial) -> Double {
        guard x.count == trainX[0].count else { return .nan }
        if let v = Loess.extrapolatedValue(x, trainX: trainX, fittedValues: fittedValues,
                                           policy: extrapolation) { return v }
        let k = Self.neighborhoodCount(n: trainX.count, span: span)
        // Single-shot query: skip the tree build (see NeighborSearch).
        let search = NeighborSearch(trainX: trainX, forBatchUse: false)
        return Loess.localFit(search: search, trainY: trainY, degree: 0,
                              at: x, neighborhood: k, robust: weights).value
    }

    /// Predictions over many queries (one shared neighbor index — much
    /// cheaper than looping `predict(_:)`).
    public func predict(_ xs: [[Double]], extrapolation: ExtrapolationPolicy = .polynomial) -> [Double] {
        let k = Self.neighborhoodCount(n: trainX.count, span: span)
        let search = NeighborSearch(trainX: trainX)
        let box = BoundingBox(trainX)
        let trainX = trainX
        let trainY = trainY
        let weights = weights
        let fittedValues = fittedValues
        let policy = extrapolation
        return xs.map { x in
            guard x.count == trainX[0].count else { return .nan }
            if !box.contains(x) {
                switch policy {
                case .polynomial:
                    break
                case .nearest:
                    guard let j = search.nearest(to: x, count: 1).first else { return .nan }
                    return fittedValues[j]
                case .unavailable:
                    return .nan
                }
            }
            return Loess.localFit(search: search, trainY: trainY, degree: 0,
                                  at: x, neighborhood: k, robust: weights).value
        }
    }

    /// Concurrent batch predictions (identical to `predict(_:)`).
    public func predictConcurrently(_ xs: [[Double]],
                                    extrapolation: ExtrapolationPolicy = .polynomial) async throws -> [Double] {
        let k = Self.neighborhoodCount(n: trainX.count, span: span)
        let search = NeighborSearch(trainX: trainX)
        let box = BoundingBox(trainX)
        let trainX = trainX
        let trainY = trainY
        let weights = weights
        let fittedValues = fittedValues
        let policy = extrapolation
        return try await concurrentMap(over: xs.count) { i in
            let x = xs[i]
            guard x.count == trainX[0].count else { return .nan }
            if !box.contains(x) {
                switch policy {
                case .polynomial:
                    break
                case .nearest:
                    guard let j = search.nearest(to: x, count: 1).first else { return .nan }
                    return fittedValues[j]
                case .unavailable:
                    return .nan
                }
            }
            return Loess.localFit(search: search, trainY: trainY, degree: 0,
                                  at: x, neighborhood: k, robust: weights).value
        }
    }

    /// Gradient ∇ŷ(x) of the kernel mean, treating robust weights as
    /// constants: dŷ/dxⱼ = Σᵢ (dWᵢ/dxⱼ)(yᵢ−ŷ)/ΣW with tricube derivatives,
    /// INCLUDING bandwidth variation (h is the max neighbor distance, so
    /// dh/dx comes from the argmax neighbor). Omitting it errs by O(1) —
    /// measured, not assumed. The span neighborhood makes ŷ piecewise
    /// smooth (kinks where membership or the argmax switches, e.g. queries
    /// exactly centered on symmetric training grids); at a kink the
    /// returned branch follows the first-maximizer rule, like ReLU at 0.
    /// Nil for empty/degenerate queries (coincident neighborhood, no
    /// positive weight) and width mismatches — mirroring the smoother's
    /// nil-instead-of-trap contract.
    public func gradient(at x: [Double]) -> [Double]? {
        guard x.count == trainX[0].count, !x.isEmpty,
              x.allSatisfy({ $0.isFinite }) else { return nil }
        let k = Self.neighborhoodCount(n: trainX.count, span: span)
        let search = NeighborSearch(trainX: trainX, forBatchUse: false)
        return Self.gradientAt(search: search, trainY: trainY, weights: weights, at: x, neighborhood: k)
    }

    /// Shared gradient kernel (single, batch, concurrent).
    static func gradientAt(search: NeighborSearch, trainY: [Double], weights: [Double],
                           at x: [Double], neighborhood k: Int) -> [Double]? {
        let trainX = search.trainingPoints
        let nb = search.nearest(to: x, count: k)
        let h = Loess.bandwidth(trainX: trainX, indices: nb, at: x)
        guard h > 0, h.isFinite else { return nil }
        let p = x.count
        // Argmax neighbor sets dh/dx (first maximizer wins ties,
        // deterministically).
        var argmax = nb.first ?? 0
        var maxD = -Double.infinity
        for j in nb {
            let d = sqrt(zip(trainX[j], x).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) })
            if d > maxD {
                maxD = d
                argmax = j
            }
        }
        let hGrad: [Double] = {
            guard maxD > 0 else { return [Double](repeating: 0, count: p) }
            return (0..<p).map { (x[$0] - trainX[argmax][$0]) / maxD }
        }()
        var total = 0.0
        var weighted = 0.0
        // w = T(u)^3, T = 1−u^3, u = d/h:
        // dw/dd = −9u²T²/h, dw/dh = 9u²T²d/h² = −(d/h)·dw/dd.
        var dwdx: [[Double]] = []
        dwdx.reserveCapacity(nb.count)
        for j in nb {
            let diff = zip(trainX[j], x).map { $0 - $1 }
            let d = sqrt(diff.reduce(0.0) { $0 + $1 * $1 })
            let u = d / h
            var dw: [Double] = [Double](repeating: 0, count: p)
            var w = 0.0
            if u > 0, u < 1 {
                let t = 1 - u * u * u
                w = t * t * t
                let dwdd = -9 * u * u * t * t / h
                let dwdh = -(d / h) * dwdd
                for m in 0..<p {
                    let dddx = d > 0 ? -diff[m] / d : 0
                    dw[m] = dwdd * dddx + dwdh * hGrad[m]
                }
            } else if u == 0 {
                w = 1
            }
            let ww = w * weights[j]
            guard ww > 0 else {
                dwdx.append([Double](repeating: 0, count: p))
                continue
            }
            total += ww
            weighted += ww * trainY[j]
            dwdx.append(dw.map { $0 * weights[j] })
        }
        guard total > 0 else { return nil }
        let mean = weighted / total
        var grad = [Double](repeating: 0, count: p)
        for (r, j) in nb.enumerated() {
            for m in 0..<p {
                grad[m] += dwdx[r][m] * (trainY[j] - mean) / total
            }
        }
        return grad
    }

    /// Gradients over many queries (one shared neighbor index).
    public func gradients(at xs: [[Double]]) -> [[Double]?] {
        let k = Self.neighborhoodCount(n: trainX.count, span: span)
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let trainY = trainY
        let weights = weights
        return xs.map { x in
            guard x.count == trainX[0].count, !x.isEmpty,
                  x.allSatisfy({ $0.isFinite }) else { return nil }
            return Self.gradientAt(search: search, trainY: trainY, weights: weights,
                                   at: x, neighborhood: k)
        }
    }

    /// Concurrent batch gradients (identical to `gradients(at:)`).
    public func gradientsConcurrently(at xs: [[Double]]) async throws -> [[Double]?] {
        let k = Self.neighborhoodCount(n: trainX.count, span: span)
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let trainY = trainY
        let weights = weights
        return try await concurrentMap(over: xs.count) { i in
            let x = xs[i]
            guard x.count == trainX[0].count, !x.isEmpty,
                  x.allSatisfy({ $0.isFinite }) else { return nil }
            return Self.gradientAt(search: search, trainY: trainY, weights: weights,
                                   at: x, neighborhood: k)
        }
    }

    /// Standard error at `x` over the span neighborhood (equivalent-kernel
    /// form, mirroring `Loess`). Nil on width mismatch or where the
    /// neighborhood solve is unavailable.
    public func standardError(at x: [Double],
                              extrapolation: ExtrapolationPolicy = .polynomial) -> Double? {
        guard x.count == trainX[0].count else { return nil }
        let k = Self.neighborhoodCount(n: trainX.count, span: span)
        // Single-shot query: skip the tree build (see NeighborSearch).
        let search = NeighborSearch(trainX: trainX, forBatchUse: false)
        let query = Loess.policyQuery(x, trainX: trainX, search: search, policy: extrapolation)
        guard let q = query else { return nil }
        return Loess.kernelStandardError(search: search, sigma: sigma, degree: 0,
                                         at: q, neighborhood: k)
    }

    /// Standard errors over many queries (one shared neighbor index).
    /// Entries are nil exactly where `standardError(at:)` is nil.
    public func standardErrors(at xs: [[Double]],
                               extrapolation: ExtrapolationPolicy = .polynomial) -> [Double?] {
        let k = Self.neighborhoodCount(n: trainX.count, span: span)
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let sigma = sigma
        let policy = extrapolation
        return xs.map { x in
            guard x.count == trainX[0].count else { return nil }
            guard let q = Loess.policyQuery(x, trainX: trainX, search: search,
                                            policy: policy) else { return nil }
            return Loess.kernelStandardError(search: search, sigma: sigma, degree: 0,
                                             at: q, neighborhood: k)
        }
    }

    /// Concurrent batch standard errors (identical to `standardErrors(at:)`).
    public func standardErrorsConcurrently(at xs: [[Double]],
                                           extrapolation: ExtrapolationPolicy = .polynomial) async throws -> [Double?] {
        let k = Self.neighborhoodCount(n: trainX.count, span: span)
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let sigma = sigma
        let policy = extrapolation
        return try await concurrentMap(over: xs.count) { i in
            let x = xs[i]
            guard x.count == trainX[0].count else { return nil }
            guard let q = Loess.policyQuery(x, trainX: trainX, search: search,
                                            policy: policy) else { return nil }
            return Loess.kernelStandardError(search: search, sigma: sigma, degree: 0,
                                             at: q, neighborhood: k)
        }
    }

    /// GCV score over candidate spans (uses each fit's trace).
    public static func selectSpan(trainX: [[Double]], trainY: [Double],
                                  spans: [Double], robustIterations: Int = 4,
                                  droppingMissing: Bool = false) -> (span: Double, fit: NadarayaWatson)? {
        var best: (span: Double, fit: NadarayaWatson)?
        var bestScore = Double.infinity
        for span in spans {
            guard let fit = NadarayaWatson.fit(trainX: trainX, trainY: trainY, span: span,
                                               robustIterations: robustIterations,
                                               droppingMissing: droppingMissing) else { continue }
            // Score on the fit's own (possibly dropped) rows (see DECISIONS #20).
            let n = Double(fit.trainY.count)
            let rss = zip(fit.trainY, fit.fittedValues).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
            let denom = max(1 - fit.trace / n, 1e-6)
            let score = (rss / n) / (denom * denom)
            if score < bestScore { bestScore = score; best = (span, fit) }
        }
        return best
    }
}
