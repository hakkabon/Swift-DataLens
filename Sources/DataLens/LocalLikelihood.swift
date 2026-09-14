import Foundation

/// Likelihood family for local-likelihood smoothing.
public enum LocalLikelihoodFamily: Sendable {
    /// Identity link, constant variance (reproduces `Loess` without robustness).
    case gaussian
    /// Logit link, Bernoulli variance μ(1−μ); responses must be 0/1.
    case binomial
    /// Log link, Poisson variance μ; responses must be ≥ 0.
    case poisson
}

/// Local-likelihood smoothing (clean-room, Loader-style): at each fit point,
/// maximize the locality-weighted log-likelihood over a local polynomial
/// for the linear predictor, via Newton–IRLS with step-halving on the local
/// deviance. The Newton systems (XᵀWX, SPD by construction) go through
/// `Regression.solveSPD` — the consumer that seam was added for.
///
/// Separated neighborhoods (e.g. all-0/all-1 binomial) have boundary MLEs:
/// the internal η scale is clamped at ±30, so fits saturate finitely instead
/// of diverging. Rank-deficient neighborhoods fall back to a bounded
/// locality-weighted mean, mirroring `Loess.localFit`. No robustness
/// reweighting rounds: likelihood families carry their own variance
/// structure (reweighting is future work — see `docs/DECISIONS.md`).
public struct LocalLikelihood: Sendable {
    public let trainX: [[Double]]
    public let trainY: [Double]
    public let degree: Int
    public let family: LocalLikelihoodFamily
    /// Span (neighborhood fraction) the fit ran under.
    public let span: Double
    /// Fitted means μ̂ = g⁻¹(η̂) at the training points.
    public let fittedValues: [Double]
    /// Fitted linear predictors η̂ at the training points.
    public let linearPredictors: [Double]
    /// Residual scale (Gaussian) or 1.0 (binomial/poisson, fixed dispersion).
    public let sigma: Double
    /// Working-influence trace Σᵢ lᵢ (effective degrees of freedom).
    public let trace: Double
    /// Global deviance Σᵢ d(yᵢ, μ̂ᵢ) at the fitted means.
    public let deviance: Double
    /// Global deviance at the intercept-only (global mean) fit.
    public let nullDeviance: Double
    /// Original input row indices kept after missing-data dropping
    /// (identity when nothing was dropped).
    public let keptIndices: [Int]

    private init(trainX: [[Double]], trainY: [Double], degree: Int, family: LocalLikelihoodFamily,
                 span: Double,
                 fittedValues: [Double], linearPredictors: [Double], sigma: Double,
                 trace: Double, deviance: Double, nullDeviance: Double,
                 keptIndices: [Int]) {
        self.trainX = trainX
        self.trainY = trainY
        self.degree = degree
        self.family = family
        self.span = span
        self.fittedValues = fittedValues
        self.linearPredictors = linearPredictors
        self.sigma = sigma
        self.trace = trace
        self.deviance = deviance
        self.nullDeviance = nullDeviance
        self.keptIndices = keptIndices
    }

    /// Monomial basis size for `degree` in `p` dimensions.
    static func basisSize(degree: Int, dimensions p: Int) -> Int {
        1 + (degree >= 1 ? p : 0) + (degree >= 2 ? p * (p + 1) / 2 : 0)
    }

    // MARK: - Family specifics (link, variance, deviance)

    /// Inverse link g⁻¹(η) for reporting (unclamped; may saturate exactly).
    static func mean(linkEta eta: Double, family: LocalLikelihoodFamily) -> Double {
        switch family {
        case .gaussian:
            return eta
        case .binomial:
            return eta >= 0 ? 1 / (1 + exp(-eta)) : exp(eta) / (1 + exp(eta))
        case .poisson:
            return exp(eta)
        }
    }

    /// Internal η clamp (±30) for weight/deviance arithmetic. Keeps exp/V
    /// finite on separation paths; the likelihood is flat there anyway.
    static func clampedEta(_ eta: Double) -> Double {
        min(max(eta, -30), 30)
    }

    /// Working triple (μ, dμ/dη, V(μ)) at clamped η for IRLS weights.
    static func workingPoint(eta: Double, family: LocalLikelihoodFamily)
        -> (mu: Double, dmu: Double, variance: Double) {
        let e = clampedEta(eta)
        switch family {
        case .gaussian:
            return (e, 1, 1)
        case .binomial:
            let mu = min(max(mean(linkEta: e, family: .binomial), 1e-12), 1 - 1e-12)
            let d = mu * (1 - mu)
            return (mu, d, d)
        case .poisson:
            let mu = max(exp(e), 1e-12)
            return (mu, mu, mu)
        }
    }

    /// Unit deviance d(y, μ) ≥ 0 (0·log0 taken as 0). μ is stability-clamped
    /// for the logarithm; fitted values themselves are reported raw.
    static func unitDeviance(y: Double, mu: Double, family: LocalLikelihoodFamily) -> Double {
        switch family {
        case .gaussian:
            let r = y - mu
            return r * r
        case .binomial:
            let m = min(max(mu, 1e-12), 1 - 1e-12)
            var d = 0.0
            if y > 0 { d += y * log(y / m) }
            if y < 1 { d += (1 - y) * log((1 - y) / (1 - m)) }
            return 2 * d
        case .poisson:
            if y == 0 { return 2 * mu }
            let m = max(mu, 1e-12)
            return 2 * (y * log(y / m) - (y - m))
        }
    }

    /// Intercept-only start (global mean through the link, clipped).
    static func startEta(trainY: [Double], family: LocalLikelihoodFamily) -> Double {
        let m = trainY.reduce(0.0, +) / Double(trainY.count)
        switch family {
        case .gaussian:
            return m
        case .binomial:
            let p = min(max(m, 1e-6), 1 - 1e-6)
            return log(p / (1 - p))
        case .poisson:
            return log(max(m, 0.1))
        }
    }

    /// Link of a fallback mean (clipped to the link's domain).
    static func fallbackEta(mean m: Double, family: LocalLikelihoodFamily) -> Double {
        switch family {
        case .gaussian:
            return m
        case .binomial:
            let p = min(max(m, 1e-6), 1 - 1e-6)
            return log(p / (1 - p))
        case .poisson:
            return log(max(m, 1e-6))
        }
    }

    // MARK: - Newton–IRLS core

    /// Result of one local IRLS fit: coefficients, normal-equations matrix
    /// at the solution (for trace/SE), and the fitted mean/eta at center.
    struct LocalFit: Sendable {
        let beta: [Double]
        let normalEquations: [[Double]]
        let mu: Double
        let eta: Double
    }

    /// Maximize Σ lw·ℓ(y; βᵀb(x)) by Newton–IRLS (≤25 iterations, step-halving
    /// on the weighted local deviance, 1e-8 relative tolerance). Returns nil
    /// on singular systems (caller falls back to a bounded local mean).
    static func localIRLS(rows: [[Double]], values: [Double], locality: [Double],
                          family: LocalLikelihoodFamily, startBeta: [Double]) -> LocalFit? {
        let n = rows.count
        guard n > 0, values.count == n, locality.count == n,
              startBeta.count == rows[0].count else { return nil }
        let q = startBeta.count
        func deviance(_ beta: [Double]) -> Double {
            var d = 0.0
            for (row, (y, lw)) in zip(rows, zip(values, locality)) {
                let eta = zip(row, beta).reduce(0.0) { $0 + $1.0 * $1.1 }
                let mu = workingPoint(eta: eta, family: family).mu
                d += lw * unitDeviance(y: y, mu: mu, family: family)
            }
            return d
        }
        var beta = startBeta
        var dev = deviance(beta)
        var converged = false
        for _ in 0..<25 {
            // Assemble BᵀWB and BᵀWz at the current beta.
            var xtx = [[Double]](repeating: [Double](repeating: 0, count: q), count: q)
            var xtz = [Double](repeating: 0, count: q)
            for (row, (y, lw)) in zip(rows, zip(values, locality)) {
                let eta = zip(row, beta).reduce(0.0) { $0 + $1.0 * $1.1 }
                let wp = workingPoint(eta: eta, family: family)
                let w = lw * wp.dmu * wp.dmu / wp.variance
                let z = eta + (y - wp.mu) / wp.dmu
                for a in 0..<q {
                    xtz[a] += row[a] * w * z
                    for b in 0..<q { xtx[a][b] += row[a] * w * row[b] }
                }
            }
            guard let candidate = Regression.solveSPD(xtx, xtz) else { return nil }
            // Step-halving on the deviance.
            var step = 1.0
            var improved = false
            var trial = beta
            var trialDev = dev
            while step >= 1.0 / 1024 {
                trial = zip(beta, candidate).map { $0 + step * ($1 - $0) }
                trialDev = deviance(trial)
                if trialDev < dev { improved = true; break }
                step /= 2
            }
            if !improved {
                converged = true // Numerical floor: deviance cannot decrease further.
                break
            }
            if abs(dev - trialDev) <= 1e-8 * max(1, abs(dev)) { converged = true }
            beta = trial
            dev = trialDev
            if converged { break }
        }
        guard converged else { return nil }
        // Normal equations at the solution (exact, for trace/SE).
        var xtx = [[Double]](repeating: [Double](repeating: 0, count: q), count: q)
        for (row, (y, lw)) in zip(rows, zip(values, locality)) {
            _ = y
            let eta = zip(row, beta).reduce(0.0) { $0 + $1.0 * $1.1 }
            let wp = workingPoint(eta: eta, family: family)
            let w = lw * wp.dmu * wp.dmu / wp.variance
            for a in 0..<q {
                for b in 0..<q { xtx[a][b] += row[a] * w * row[b] }
            }
        }
        let eta0 = beta[0]
        return LocalFit(beta: beta, normalEquations: xtx,
                        mu: mean(linkEta: eta0, family: family), eta: eta0)
    }

    /// One local fit at `x` over `k` neighbors. Falls back to a bounded
    /// locality-weighted mean (leverage 0, no equations) on degeneracy.
    struct PointFit: Sendable {
        let mu: Double
        let eta: Double
        let coefficients: [Double]?
        let normalEquations: [[Double]]?
        let ownWeight: Double
        let bandwidth: Double
    }

    static func localFit(search: NeighborSearch, trainY: [Double], degree: Int,
                         family: LocalLikelihoodFamily, at x: [Double],
                         neighborhood k: Int, startBeta: [Double]) -> PointFit? {
        let trainX = search.trainingPoints
        let nb = search.nearest(to: x, count: k)
        guard !nb.isEmpty else { return nil }
        let h = Loess.bandwidth(trainX: trainX, indices: nb, at: x)
        var rows: [[Double]] = [], vals: [Double] = [], lw: [Double] = []
        for j in nb {
            let d = sqrt(zip(trainX[j], x).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) })
            rows.append(Loess.basis(trainX[j], center: x, degree: degree))
            vals.append(trainY[j])
            lw.append(h <= 0 ? 1 : LoessWeight.tricube(d / h))
        }
        guard let r = localIRLS(rows: rows, values: vals, locality: lw,
                                family: family, startBeta: startBeta) else {
            // Bounded local mean (mirrors Loess.localFit's fallback).
            let total = lw.reduce(0.0, +)
            if total > 0 {
                let m = zip(vals, lw).reduce(0.0) { $0 + $1.0 * $1.1 } / total
                return PointFit(mu: m, eta: fallbackEta(mean: m, family: family),
                                coefficients: nil, normalEquations: nil, ownWeight: 0, bandwidth: h)
            }
            let m = trainY[nb.first ?? 0]
            return PointFit(mu: m, eta: fallbackEta(mean: m, family: family),
                            coefficients: nil, normalEquations: nil, ownWeight: 0, bandwidth: h)
        }
        // Own working weight: self sits at distance 0 (locality 1).
        let wp = workingPoint(eta: r.eta, family: family)
        let ownWeight = wp.dmu * wp.dmu / wp.variance
        return PointFit(mu: r.mu, eta: r.eta, coefficients: r.beta,
                        normalEquations: r.normalEquations,
                        ownWeight: ownWeight, bandwidth: h)
    }

    /// Working-influence leverage lᵢ = wᵢ·(M⁻¹)[0][0] (own basis row is e₁
    /// by centering). Zero when the point fell back (no equations).
    static func leverage(normalEquations: [[Double]]?, ownWeight: Double) -> Double {
        guard let m = normalEquations, !m.isEmpty else { return 0 }
        var e1 = [Double](repeating: 0, count: m.count)
        e1[0] = 1
        guard let col = Regression.solveSPD(m, e1) else { return 0 }
        return ownWeight * col[0]
    }

    // MARK: - Public API

    /// Fit by local likelihood under `family` with a fixed `span`
    /// neighborhood fraction (mirroring `Loess.fit`'s span semantics).
    /// Binomial responses must be 0/1; Poisson responses must be ≥ 0.
    public static func fit(trainX: [[Double]], trainY: [Double],
                           degree: Int = 2, family: LocalLikelihoodFamily = .gaussian,
                           span: Double = 0.75,
                           droppingMissing: Bool = false) -> LocalLikelihood? {
        guard trainX.count == trainY.count else { return nil }
        let (trainX, trainY, keptIndices): ([[Double]], [Double], [Int]) = droppingMissing
            ? MissingData.dropping(trainX: trainX, trainY: trainY)
            : (trainX, trainY, Array(trainX.indices))
        guard !trainX.isEmpty,
              span > 0, span <= 1, (0...2).contains(degree),
              trainX.allSatisfy({ $0.count == trainX[0].count }),
              trainX.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              trainY.allSatisfy({ $0.isFinite }) else { return nil }
        switch family {
        case .binomial:
            guard trainY.allSatisfy({ $0 == 0 || $0 == 1 }) else { return nil }
        case .poisson:
            guard trainY.allSatisfy({ $0 >= 0 }) else { return nil }
        case .gaussian:
            break
        }
        let n = trainX.count
        let p = trainX[0].count
        let q = basisSize(degree: degree, dimensions: p)
        let k = min(n, max(Int(ceil(span * Double(n))), q + 1))
        guard k > 1 else { return nil }
        let search = NeighborSearch(trainX: trainX)
        let eta0 = startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        var fitted = [Double](repeating: 0, count: n)
        var etas = [Double](repeating: 0, count: n)
        var trace = 0.0
        for i in 0..<n {
            guard let r = localFit(search: search, trainY: trainY, degree: degree,
                                   family: family, at: trainX[i],
                                   neighborhood: k, startBeta: startBeta) else { return nil }
            fitted[i] = r.mu
            etas[i] = r.eta
            trace += leverage(normalEquations: r.normalEquations, ownWeight: r.ownWeight)
        }
        let rss = zip(trainY, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma: Double
        switch family {
        case .gaussian:
            sigma = sqrt(rss / max(Double(n) - trace, 1))
        case .binomial, .poisson:
            sigma = 1.0
        }
        let deviance = zip(trainY, fitted).reduce(0.0) {
            $0 + unitDeviance(y: $1.0, mu: $1.1, family: family)
        }
        let nullMu: Double
        switch family {
        case .gaussian:
            nullMu = trainY.reduce(0.0, +) / Double(n)
        case .binomial:
            nullMu = min(max(trainY.reduce(0.0, +) / Double(n), 1e-6), 1 - 1e-6)
        case .poisson:
            nullMu = max(trainY.reduce(0.0, +) / Double(n), 1e-6)
        }
        let nullDeviance = trainY.reduce(0.0) {
            $0 + unitDeviance(y: $1, mu: nullMu, family: family)
        }
        return LocalLikelihood(trainX: trainX, trainY: trainY, degree: degree, family: family,
                               span: span,
                               fittedValues: fitted, linearPredictors: etas, sigma: sigma,
                               trace: trace, deviance: deviance, nullDeviance: nullDeviance,
                               keptIndices: keptIndices)
    }

    /// Concurrent fit: identical to `fit(trainX:trainY:degree:family:span:)`.
    ///
    /// Per-point IRLS runs in parallel; reductions stay sequential, in index
    /// order. Bit-identical to `fit` (pinned by `BatchTests`).
    public static func fitConcurrently(trainX: [[Double]], trainY: [Double],
                                       degree: Int = 2, family: LocalLikelihoodFamily = .gaussian,
                                       span: Double = 0.75,
                                       droppingMissing: Bool = false) async -> LocalLikelihood? {
        guard trainX.count == trainY.count else { return nil }
        let (trainX, trainY, keptIndices): ([[Double]], [Double], [Int]) = droppingMissing
            ? MissingData.dropping(trainX: trainX, trainY: trainY)
            : (trainX, trainY, Array(trainX.indices))
        guard !trainX.isEmpty,
              span > 0, span <= 1, (0...2).contains(degree),
              trainX.allSatisfy({ $0.count == trainX[0].count }),
              trainX.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              trainY.allSatisfy({ $0.isFinite }) else { return nil }
        switch family {
        case .binomial:
            guard trainY.allSatisfy({ $0 == 0 || $0 == 1 }) else { return nil }
        case .poisson:
            guard trainY.allSatisfy({ $0 >= 0 }) else { return nil }
        case .gaussian:
            break
        }
        let n = trainX.count
        let p = trainX[0].count
        let q = basisSize(degree: degree, dimensions: p)
        let k = min(n, max(Int(ceil(span * Double(n))), q + 1))
        guard k > 1 else { return nil }
        let search = NeighborSearch(trainX: trainX)
        let eta0 = startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        typealias PointOut = (mu: Double, eta: Double, xtx: [[Double]]?,
                              ownWeight: Double, bandwidth: Double)
        let points: [PointOut?] = await concurrentMap(over: n) { i in
            guard let r = localFit(search: search, trainY: trainY, degree: degree,
                                   family: family, at: trainX[i],
                                   neighborhood: k, startBeta: startBeta) else { return nil }
            return (r.mu, r.eta, r.normalEquations, r.ownWeight, r.bandwidth)
        }
        var fitted = [Double](repeating: 0, count: n)
        var etas = [Double](repeating: 0, count: n)
        var trace = 0.0
        for (i, r) in points.enumerated() {
            guard let r else { return nil }
            fitted[i] = r.mu
            etas[i] = r.eta
            trace += leverage(normalEquations: r.xtx, ownWeight: r.ownWeight)
        }
        let rss = zip(trainY, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma: Double
        switch family {
        case .gaussian:
            sigma = sqrt(rss / max(Double(n) - trace, 1))
        case .binomial, .poisson:
            sigma = 1.0
        }
        let deviance = zip(trainY, fitted).reduce(0.0) {
            $0 + unitDeviance(y: $1.0, mu: $1.1, family: family)
        }
        let nullMu: Double
        switch family {
        case .gaussian:
            nullMu = trainY.reduce(0.0, +) / Double(n)
        case .binomial:
            nullMu = min(max(trainY.reduce(0.0, +) / Double(n), 1e-6), 1 - 1e-6)
        case .poisson:
            nullMu = max(trainY.reduce(0.0, +) / Double(n), 1e-6)
        }
        let nullDeviance = trainY.reduce(0.0) {
            $0 + unitDeviance(y: $1, mu: nullMu, family: family)
        }
        return LocalLikelihood(trainX: trainX, trainY: trainY, degree: degree, family: family,
                               span: span,
                               fittedValues: fitted, linearPredictors: etas, sigma: sigma,
                               trace: trace, deviance: deviance, nullDeviance: nullDeviance,
                               keptIndices: keptIndices)
    }

    /// Predict the mean at `x` (`.nan` on width mismatch).
    public func predict(_ x: [Double], extrapolation: ExtrapolationPolicy = .polynomial) -> Double {
        guard x.count == trainX[0].count else { return .nan }
        let q = LocalLikelihood.basisSize(degree: degree, dimensions: x.count)
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), q + 1))
        let search = NeighborSearch(trainX: trainX, forBatchUse: false)
        let eta0 = LocalLikelihood.startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        return LocalLikelihood.predictAt(search: search, trainX: trainX, trainY: trainY, degree: degree,
                                         family: family, fittedValues: fittedValues,
                                         at: x, neighborhood: k, startBeta: startBeta,
                                         policy: extrapolation)
    }

    /// Shared per-point prediction kernel (single, batch, and concurrent
    /// evaluation all funnel through here).
    static func predictAt(search: NeighborSearch, trainX: [[Double]], trainY: [Double], degree: Int,
                          family: LocalLikelihoodFamily, fittedValues: [Double],
                          at x: [Double], neighborhood k: Int, startBeta: [Double],
                          policy: ExtrapolationPolicy) -> Double {
        if !BoundingBox(trainX).contains(x) {
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
        guard let r = localFit(search: search, trainY: trainY, degree: degree,
                               family: family, at: x,
                               neighborhood: k, startBeta: startBeta) else { return .nan }
        return r.mu
    }

    /// Predictions over many queries (one shared neighbor index).
    public func predict(_ xs: [[Double]], extrapolation: ExtrapolationPolicy = .polynomial) -> [Double] {
        let p = trainX[0].count
        let q = LocalLikelihood.basisSize(degree: degree, dimensions: p)
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), q + 1))
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let trainY = trainY
        let degree = degree
        let family = family
        let fittedValues = fittedValues
        let eta0 = LocalLikelihood.startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        let policy = extrapolation
        return xs.map { x in
            guard x.count == p else { return .nan }
            return LocalLikelihood.predictAt(search: search, trainX: trainX, trainY: trainY,
                                             degree: degree, family: family, fittedValues: fittedValues,
                                             at: x, neighborhood: k, startBeta: startBeta,
                                             policy: policy)
        }
    }

    /// Concurrent batch predictions (identical to `predict(_:)`).
    public func predictConcurrently(_ xs: [[Double]],
                                    extrapolation: ExtrapolationPolicy = .polynomial) async -> [Double] {
        let p = trainX[0].count
        let q = LocalLikelihood.basisSize(degree: degree, dimensions: p)
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), q + 1))
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let trainY = trainY
        let degree = degree
        let family = family
        let fittedValues = fittedValues
        let eta0 = LocalLikelihood.startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        let policy = extrapolation
        return await concurrentMap(over: xs.count) { i in
            let x = xs[i]
            guard x.count == p else { return .nan }
            return LocalLikelihood.predictAt(search: search, trainX: trainX, trainY: trainY,
                                             degree: degree, family: family, fittedValues: fittedValues,
                                             at: x, neighborhood: k, startBeta: startBeta,
                                             policy: policy)
        }
    }

    /// Delta-method standard error se(μ̂) = |dμ/dη|·√(e₁ᵀM⁻¹e₁) at the
    /// solution (nil on width mismatch or degenerate neighborhoods).
    public func standardError(at x: [Double],
                              extrapolation: ExtrapolationPolicy = .polynomial) -> Double? {
        guard x.count == trainX[0].count else { return nil }
        let q = LocalLikelihood.basisSize(degree: degree, dimensions: x.count)
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), q + 1))
        let search = NeighborSearch(trainX: trainX, forBatchUse: false)
        let eta0 = LocalLikelihood.startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        return LocalLikelihood.standardErrorAt(search: search, trainX: trainX, trainY: trainY,
                                               degree: degree, family: family,
                                               at: x, neighborhood: k, startBeta: startBeta,
                                               policy: extrapolation)
    }

    /// Shared per-point SE kernel.
    static func standardErrorAt(search: NeighborSearch, trainX: [[Double]], trainY: [Double], degree: Int,
                                family: LocalLikelihoodFamily,
                                at x: [Double], neighborhood k: Int, startBeta: [Double],
                                policy: ExtrapolationPolicy) -> Double? {
        let query: [Double]
        switch policy {
        case .polynomial:
            query = x
        case .nearest:
            guard BoundingBox(trainX).contains(x) else {
                guard let j = search.nearest(to: x, count: 1).first else { return nil }
                query = trainX[j]
                break
            }
            query = x
        case .unavailable:
            guard BoundingBox(trainX).contains(x) else { return nil }
            query = x
        }
        guard let r = localFit(search: search, trainY: trainY, degree: degree,
                               family: family, at: query,
                               neighborhood: k, startBeta: startBeta),
            let m = r.normalEquations, !m.isEmpty
        else { return nil }
        var e1 = [Double](repeating: 0, count: m.count)
        e1[0] = 1
        guard let col = Regression.solveSPD(m, e1) else { return nil }
        let wp = workingPoint(eta: r.eta, family: family)
        return abs(wp.dmu) * sqrt(max(col[0], 0))
    }

    /// Gradient of the fitted mean ∇μ̂(x) = dμ/dη · β[1...p] from the local
    /// coefficients. Nil for degree-0 fits and degenerate neighborhoods;
    /// local-polynomial by construction (ignores the extrapolation policy).
    static func gradientAt(search: NeighborSearch, trainY: [Double], degree: Int,
                           family: LocalLikelihoodFamily, at x: [Double],
                           neighborhood k: Int, startBeta: [Double]) -> [Double]? {
        guard degree >= 1, !x.isEmpty else { return nil }
        guard let r = localFit(search: search, trainY: trainY, degree: degree,
                               family: family, at: x,
                               neighborhood: k, startBeta: startBeta),
            let beta = r.coefficients, beta.count >= 1 + x.count
        else { return nil }
        let wp = workingPoint(eta: r.eta, family: family)
        return beta[1...x.count].map { wp.dmu * $0 }
    }

    /// Gradient of the fitted mean ∇μ̂(x) = dμ/dη · β[1...p] from the local
    /// coefficients. Nil for degree-0 fits and degenerate neighborhoods;
    /// local-polynomial by construction (ignores the extrapolation policy).
    public func gradient(at x: [Double]) -> [Double]? {
        guard x.count == trainX[0].count, !x.isEmpty else { return nil }
        let q = LocalLikelihood.basisSize(degree: degree, dimensions: x.count)
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), q + 1))
        let search = NeighborSearch(trainX: trainX, forBatchUse: false)
        let eta0 = LocalLikelihood.startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        return LocalLikelihood.gradientAt(search: search, trainY: trainY, degree: degree,
                                          family: family, at: x,
                                          neighborhood: k, startBeta: startBeta)
    }

    /// Gradients over many queries (one shared neighbor index).
    public func gradients(at xs: [[Double]]) -> [[Double]?] {
        let p = trainX[0].count
        guard p > 0 else { return xs.map { _ in nil } }
        let q = LocalLikelihood.basisSize(degree: degree, dimensions: p)
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), q + 1))
        let search = NeighborSearch(trainX: trainX)
        let trainY = trainY
        let degree = degree
        let family = family
        let eta0 = LocalLikelihood.startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        return xs.map { x in
            guard x.count == p else { return nil }
            return LocalLikelihood.gradientAt(search: search, trainY: trainY, degree: degree,
                                              family: family, at: x,
                                              neighborhood: k, startBeta: startBeta)
        }
    }

    /// Concurrent batch gradients (identical to `gradients(at:)`).
    public func gradientsConcurrently(at xs: [[Double]]) async -> [[Double]?] {
        let p = trainX[0].count
        guard p > 0 else { return xs.map { _ in nil } }
        let q = LocalLikelihood.basisSize(degree: degree, dimensions: p)
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), q + 1))
        let search = NeighborSearch(trainX: trainX)
        let trainY = trainY
        let degree = degree
        let family = family
        let eta0 = LocalLikelihood.startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        return await concurrentMap(over: xs.count) { i in
            let x = xs[i]
            guard x.count == p else { return nil }
            return LocalLikelihood.gradientAt(search: search, trainY: trainY, degree: degree,
                                              family: family, at: x,
                                              neighborhood: k, startBeta: startBeta)
        }
    }

    /// Standard errors over many queries (one shared neighbor index).
    public func standardErrors(at xs: [[Double]],
                               extrapolation: ExtrapolationPolicy = .polynomial) -> [Double?] {
        let p = trainX[0].count
        let q = LocalLikelihood.basisSize(degree: degree, dimensions: p)
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), q + 1))
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let trainY = trainY
        let degree = degree
        let family = family
        let eta0 = LocalLikelihood.startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        let policy = extrapolation
        return xs.map { x in
            guard x.count == p else { return nil }
            return LocalLikelihood.standardErrorAt(search: search, trainX: trainX, trainY: trainY,
                                                   degree: degree, family: family,
                                                   at: x, neighborhood: k, startBeta: startBeta,
                                                   policy: policy)
        }
    }

    /// Concurrent batch standard errors (identical to `standardErrors(at:)`).
    public func standardErrorsConcurrently(at xs: [[Double]],
                                           extrapolation: ExtrapolationPolicy = .polynomial) async -> [Double?] {
        let p = trainX[0].count
        let q = LocalLikelihood.basisSize(degree: degree, dimensions: p)
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), q + 1))
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let trainY = trainY
        let degree = degree
        let family = family
        let eta0 = LocalLikelihood.startEta(trainY: trainY, family: family)
        let startBeta = [eta0] + [Double](repeating: 0, count: max(q - 1, 0))
        let policy = extrapolation
        return await concurrentMap(over: xs.count) { i in
            let x = xs[i]
            guard x.count == p else { return nil }
            return LocalLikelihood.standardErrorAt(search: search, trainX: trainX, trainY: trainY,
                                                   degree: degree, family: family,
                                                   at: x, neighborhood: k, startBeta: startBeta,
                                                   policy: policy)
        }
    }

    /// AIC span selection (deviance + 2·trace) over candidate spans.
    public static func selectSpan(trainX: [[Double]], trainY: [Double],
                                  spans: [Double], degree: Int = 2,
                                  family: LocalLikelihoodFamily = .gaussian,
                                  droppingMissing: Bool = false) -> (span: Double, fit: LocalLikelihood)? {
        var best: (span: Double, fit: LocalLikelihood)?
        var bestScore = Double.infinity
        for span in spans {
            guard let fit = LocalLikelihood.fit(trainX: trainX, trainY: trainY,
                                                degree: degree, family: family,
                                                span: span,
                                                droppingMissing: droppingMissing) else { continue }
            guard fit.deviance.isFinite else { continue }
            let score = fit.deviance + 2 * fit.trace
            if score < bestScore { bestScore = score; best = (span, fit) }
        }
        return best
    }
}
