import Foundation

/// Adaptive local-polynomial smoothing: Loader-style per-point bandwidth
/// selection over Cleveland-style local fits (clean-room design, no `locfit`
/// code).
///
/// A fixed span oversmooths wiggly regions and undersmooths flat ones.
/// `AdaptiveLoess` instead scores candidate neighborhood sizes at each fit
/// point with a local corrected-Akaike criterion and keeps the minimizer:
///
///     AICc(k) = ln(RSS(k)/k) + [2q + 2q(q+1)/(k−q−1)]/k
///
/// where RSS(k) is the unweighted residual sum of squares of the local
/// degree-`degree` fit (q basis terms) over its k neighbors. Per-observation
/// on purpose (totals would let the neighborhood size dominate); exact ties
/// resolve toward larger neighborhoods. Selection runs once, unweighted
/// (mirroring how `Loess` fixes its span across rounds); refinement rounds
/// then reweight with the bisquare robustness function on the fixed
/// neighborhoods, exactly like `Loess`.
///
/// Shares `LocalPolynomial` (fits), `NeighborSearch` (neighbors),
/// `Loess/bandwidth(_:indices:at:)` and `Loess/kernelStandardError`
/// with classic `Loess`, whose behavior is unchanged.
public struct AdaptiveLoess: Sendable {
    public let trainX: [[Double]]
    public let trainY: [Double]
    public let degree: Int
    /// Fitted values at the training points (final robustness iteration).
    public let fittedValues: [Double]
    /// Residual scale σ̂ = √(RSS/max(n−tr,1)).
    public let sigma: Double
    /// Smoother-matrix trace Σᵢ lᵢ(xᵢ) (effective degrees of freedom).
    public let trace: Double
    /// Final bisquare robustness weights per training point.
    public let weights: [Double]
    /// Selected neighborhood size per training point.
    public let selectedNeighborhoods: [Int]
    /// Local bandwidth (max neighbor distance) per training point.
    public let bandwidths: [Double]
    /// Candidate grid the selection ran over (reused by predict/SE).
    public let candidateNeighborhoods: [Int]

    private init(trainX: [[Double]], trainY: [Double], degree: Int,
                 fittedValues: [Double], sigma: Double, trace: Double,
                 weights: [Double], selectedNeighborhoods: [Int], bandwidths: [Double],
                 candidateNeighborhoods: [Int]) {
        self.trainX = trainX
        self.trainY = trainY
        self.degree = degree
        self.fittedValues = fittedValues
        self.sigma = sigma
        self.trace = trace
        self.weights = weights
        self.selectedNeighborhoods = selectedNeighborhoods
        self.bandwidths = bandwidths
        self.candidateNeighborhoods = candidateNeighborhoods
    }

    /// Monomial basis size for `degree` in `p` dimensions.
    static func basisSize(degree: Int, dimensions p: Int) -> Int {
        1 + (degree >= 1 ? p : 0) + (degree >= 2 ? p * (p + 1) / 2 : 0)
    }

    /// Default candidate grid: geometrically spaced sizes from `kMin` to
    /// `n` (always ending at `n`, so flat regions may go fully global).
    /// Returns `[]` when no valid size exists (`kMin > n`).
    static func defaultGrid(n: Int, kMin: Int) -> [Int] {
        guard kMin <= n else { return [] }
        var grid = [kMin]
        while grid.count < 8, let last = grid.last, last < n {
            grid.append(min(n, max(last + 1, Int((Double(last) * 1.6).rounded()))))
        }
        if grid.last != n { grid.append(n) }
        return grid
    }

    /// Corrected-Akaike score of a local fit, per observation:
    /// ln(RSS/k) + [2q + 2q(q+1)/(k−q−1)]/k.
    ///
    /// Per-observation on purpose: totals (k·ln(RSS/k) + …) let the
    /// neighborhood SIZE dominate the comparison instead of fit quality
    /// (the k·ln(RSS/k) term grows with k whenever RSS/k > 1 and shrinks
    /// whenever it is < 1). Dividing by k keeps the bias–variance reading
    /// intact across sizes. Exact fits (RSS = 0) score −∞ and always win.
    /// Requires `k > q + 1` (callers enforce `k ≥ q+2`).
    static func aicc(residualSumOfSquares rss: Double, count k: Int, parameters q: Int) -> Double {
        let dq = Double(q)
        return log(rss / Double(k)) + (2 * dq + 2 * dq * (dq + 1) / Double(k - q - 1)) / Double(k)
    }

    /// One local evaluation at `x` over `k` neighbors: fitted value and
    /// leverage (Nadaraya–Watson / bounded-mean fallback cascade mirroring
    /// `Loess.localFit`), plus the QR coefficients and neighbor rows for
    /// scoring (`nil` coefficients when the design is rank-deficient).
    /// Returns nil only when no neighbor carries positive weight.
    struct LocalEvaluation: Sendable {
        let value: Double
        let leverage: Double
        let bandwidth: Double
        let coefficients: [Double]?
        let neighborRows: [[Double]]
        let neighborValues: [Double]
    }

    static func evaluate(search: NeighborSearch, trainY: [Double], degree: Int,
                         at x: [Double], neighborhood k: Int, robust: [Double],
                         track: Int?) -> LocalEvaluation? {
        let trainX = search.trainingPoints
        let nb = search.nearest(to: x, count: k)
        let h = Loess.bandwidth(trainX: trainX, indices: nb, at: x)
        var lwPairs: [(idx: Int, w: Double)] = []
        for j in nb {
            let d = sqrt(zip(trainX[j], x).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) })
            lwPairs.append((j, h <= 0 ? 1 : LoessWeight.tricube(d / h)))
        }
        var rows: [[Double]] = [], vals: [Double] = [], wts: [Double] = [], idx: [Int] = []
        for (j, lw) in lwPairs {
            let ww = lw * robust[j]
            if ww > 0 {
                rows.append(Loess.basis(trainX[j], center: x, degree: degree))
                vals.append(trainY[j])
                wts.append(ww)
                idx.append(j)
            }
        }
        // Degenerate combined set: bounded local mean ignoring robust
        // (mirrors Loess.localFit); unscored downstream (nil coefficients).
        guard !wts.isEmpty else {
            var frows: [[Double]] = [], fvals: [Double] = []
            var total = 0.0
            var value = 0.0
            for (j, lw) in lwPairs where lw > 0 {
                frows.append(Loess.basis(trainX[j], center: x, degree: degree))
                fvals.append(trainY[j])
                total += lw
                value += trainY[j] * lw
            }
            if total > 0 {
                value /= total
            } else {
                value = trainY[nb.first ?? 0]
            }
            return LocalEvaluation(value: value, leverage: 0, bandwidth: h,
                                   coefficients: nil, neighborRows: frows, neighborValues: fvals)
        }
        // Value/leverage cascade mirrors Loess.localFit exactly.
        var value = 0.0
        var leverage = 0.0
        if degree == 0 || wts.count == 1 {
            let total = wts.reduce(0.0, +)
            value = zip(vals, wts).reduce(0.0) { $0 + $1.0 * $1.1 } / total
            if let t = track, let p = idx.firstIndex(of: t) { leverage = wts[p] / total }
        } else {
            let pos = track.flatMap { t in idx.firstIndex(of: t) }
            guard let r = LocalPolynomial.fitWeighted(rows: rows, values: vals,
                                                      weights: wts, track: pos) else {
                let total = wts.reduce(0.0, +)
                value = zip(vals, wts).reduce(0.0) { $0 + $1.0 * $1.1 } / total
                return LocalEvaluation(value: value, leverage: 0, bandwidth: h,
                                       coefficients: nil, neighborRows: rows, neighborValues: vals)
            }
            value = r.coefficients[0]
            leverage = r.leverage
        }
        // Coefficients for scoring (intercept-only QR always succeeds here,
        // so degree-0 evaluations score too; single-point degree ≥ 1 yields nil).
        let beta = LocalPolynomial.fitWeighted(rows: rows, values: vals,
                                               weights: wts, track: nil)?.coefficients
        return LocalEvaluation(value: value, leverage: leverage, bandwidth: h,
                               coefficients: beta, neighborRows: rows, neighborValues: vals)
    }

    /// Select the AICc-minimizing neighborhood at `x` over `candidates`
    /// (ascending; exact ties resolve toward larger neighborhoods).
    /// Selection is a pure function of the data (unweighted fits).
    /// Returns nil only when every candidate is degenerate there.
    static func select(search: NeighborSearch, trainY: [Double], degree: Int, parameters q: Int,
                       at x: [Double], candidates: [Int]) -> (neighborhood: Int, bandwidth: Double)? {
        let ones = [Double](repeating: 1, count: search.trainingPoints.count)
        var best: (neighborhood: Int, bandwidth: Double)?
        var bestScore = Double.infinity
        var fallback: (neighborhood: Int, bandwidth: Double)?
        for k in candidates.sorted() {
            guard let e = evaluate(search: search, trainY: trainY, degree: degree,
                                   at: x, neighborhood: k, robust: ones, track: nil) else { continue }
            fallback = (k, e.bandwidth)
            guard let beta = e.coefficients else { continue }
            let rss = zip(e.neighborRows, e.neighborValues).reduce(0.0) { acc, pair in
                let r = pair.1 - zip(pair.0, beta).reduce(0.0) { $0 + $1.0 * $1.1 }
                return acc + r * r
            }
            let score = aicc(residualSumOfSquares: rss, count: k, parameters: q)
            if score <= bestScore {
                bestScore = score
                best = (k, e.bandwidth)
            }
        }
        // All-rank-deficient (e.g. duplicated inputs): degrade gracefully to
        // the largest neighborhood's bounded fit, mirroring Loess.
        return best ?? fallback
    }

    /// Fit with per-point AICc neighborhoods. `neighborhoods` lists candidate
    /// sizes (filtered to `q+2...n`); nil selects the default grid. Selection
    /// runs once, unweighted; `robustIterations` bisquare rounds then refine
    /// on the fixed neighborhoods (4 matches R, mirroring `Loess`).
    public static func fit(trainX: [[Double]], trainY: [Double], degree: Int = 2,
                           neighborhoods: [Int]? = nil,
                           robustIterations: Int = 4) -> AdaptiveLoess? {
        guard !trainX.isEmpty, trainX.count == trainY.count, (0...2).contains(degree),
              trainX.allSatisfy({ $0.count == trainX[0].count }),
              trainX.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              trainY.allSatisfy({ $0.isFinite }) else { return nil }
        let n = trainX.count
        let p = trainX[0].count
        let q = basisSize(degree: degree, dimensions: p)
        let kMin = q + 2
        let candidates = Array(Set(neighborhoods ?? defaultGrid(n: n, kMin: kMin))
            .filter { $0 >= kMin && $0 <= n }).sorted()
        guard !candidates.isEmpty else { return nil }
        let search = NeighborSearch(trainX: trainX)
        let ones = [Double](repeating: 1, count: n)
        // Select once (unweighted), fit round 0.
        var selected = [Int](repeating: 0, count: n)
        var fitted = [Double](repeating: 0, count: n)
        var bandwidths = [Double](repeating: 0, count: n)
        for i in 0..<n {
            guard let s = select(search: search, trainY: trainY, degree: degree, parameters: q,
                                 at: trainX[i], candidates: candidates) else { return nil }
            selected[i] = s.neighborhood
            guard let e = evaluate(search: search, trainY: trainY, degree: degree,
                                   at: trainX[i], neighborhood: s.neighborhood,
                                   robust: ones, track: nil) else { return nil }
            fitted[i] = e.value
            bandwidths[i] = e.bandwidth
        }
        // Bisquare refinement on the fixed neighborhoods (mirrors Loess,
        // including the y-scale floor against median-collapse).
        let medY = Descriptive.median(trainY) ?? 0
        let yScale = max(Descriptive.median(trainY.map { abs($0 - medY) }) ?? 0, 1e-300)
        var robust = ones
        for _ in 0..<robustIterations {
            let resid = zip(trainY, fitted).map { abs($0 - $1) }
            guard let s = Descriptive.median(resid) else { break }
            let sEff = max(s, 1e-8 * yScale)
            robust = resid.map { LoessWeight.bisquare($0 / (6 * sEff)) }
            for i in 0..<n {
                guard let e = evaluate(search: search, trainY: trainY, degree: degree,
                                       at: trainX[i], neighborhood: selected[i],
                                       robust: robust, track: nil) else { return nil }
                fitted[i] = e.value
            }
        }
        // Final pass with diagnostics.
        var trace = 0.0
        for i in 0..<n {
            guard let e = evaluate(search: search, trainY: trainY, degree: degree,
                                   at: trainX[i], neighborhood: selected[i],
                                   robust: robust, track: i) else { return nil }
            fitted[i] = e.value
            bandwidths[i] = e.bandwidth
            trace += e.leverage
        }
        let rss = zip(trainY, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma = sqrt(rss / max(Double(n) - trace, 1))
        return AdaptiveLoess(trainX: trainX, trainY: trainY, degree: degree,
                             fittedValues: fitted, sigma: sigma, trace: trace,
                             weights: robust, selectedNeighborhoods: selected,
                             bandwidths: bandwidths, candidateNeighborhoods: candidates)
    }

    /// Concurrent fit: identical to `fit(trainX:trainY:degree:neighborhoods:robustIterations:)`.
    ///
    /// Selection, refits, and the final pass run per-point in parallel;
    /// residual medians and the trace stay sequential, in index order.
    /// Bit-identical to `fit` (pinned by `BatchTests`).
    public static func fitConcurrently(trainX: [[Double]], trainY: [Double], degree: Int = 2,
                                       neighborhoods: [Int]? = nil,
                                       robustIterations: Int = 4) async -> AdaptiveLoess? {
        guard !trainX.isEmpty, trainX.count == trainY.count, (0...2).contains(degree),
              trainX.allSatisfy({ $0.count == trainX[0].count }),
              trainX.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              trainY.allSatisfy({ $0.isFinite }) else { return nil }
        let n = trainX.count
        let p = trainX[0].count
        let q = basisSize(degree: degree, dimensions: p)
        let kMin = q + 2
        let candidates = Array(Set(neighborhoods ?? defaultGrid(n: n, kMin: kMin))
            .filter { $0 >= kMin && $0 <= n }).sorted()
        guard !candidates.isEmpty else { return nil }
        let search = NeighborSearch(trainX: trainX)
        let ones = [Double](repeating: 1, count: n)
        var selected = [Int](repeating: 0, count: n)
        var fitted = [Double](repeating: 0, count: n)
        var bandwidths = [Double](repeating: 0, count: n)
        let round0 = await concurrentMap(over: n) { i -> (Int, Double, Double)? in
            guard let s = AdaptiveLoess.select(search: search, trainY: trainY, degree: degree,
                                               parameters: q, at: trainX[i],
                                               candidates: candidates) else { return nil }
            guard let e = AdaptiveLoess.evaluate(search: search, trainY: trainY, degree: degree,
                                                 at: trainX[i], neighborhood: s.neighborhood,
                                                 robust: ones, track: nil) else { return nil }
            return (s.neighborhood, e.value, e.bandwidth)
        }
        for (i, r) in round0.enumerated() {
            guard let (k, v, h) = r else { return nil }
            selected[i] = k
            fitted[i] = v
            bandwidths[i] = h
        }
        let medY = Descriptive.median(trainY) ?? 0
        let yScale = max(Descriptive.median(trainY.map { abs($0 - medY) }) ?? 0, 1e-300)
        var robust = ones
        for _ in 0..<robustIterations {
            let resid = zip(trainY, fitted).map { abs($0 - $1) }
            guard let s = Descriptive.median(resid) else { break }
            let sEff = max(s, 1e-8 * yScale)
            robust = resid.map { LoessWeight.bisquare($0 / (6 * sEff)) }
            let currentRobust = robust
            let currentSelected = selected
            let vals: [Double?] = await concurrentMap(over: n) { i in
                AdaptiveLoess.evaluate(search: search, trainY: trainY, degree: degree,
                                       at: trainX[i], neighborhood: currentSelected[i],
                                       robust: currentRobust, track: nil)?.value
            }
            for (i, v) in vals.enumerated() {
                guard let v else { return nil }
                fitted[i] = v
            }
        }
        var trace = 0.0
        let currentRobust = robust
        let currentSelected = selected
        let final = await concurrentMap(over: n) { i -> (Double, Double, Double)? in
            guard let e = AdaptiveLoess.evaluate(search: search, trainY: trainY, degree: degree,
                                                 at: trainX[i], neighborhood: currentSelected[i],
                                                 robust: currentRobust, track: i) else { return nil }
            return (e.value, e.leverage, e.bandwidth)
        }
        for (i, r) in final.enumerated() {
            guard let (v, l, h) = r else { return nil }
            fitted[i] = v
            bandwidths[i] = h
            trace += l
        }
        let rss = zip(trainY, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma = sqrt(rss / max(Double(n) - trace, 1))
        return AdaptiveLoess(trainX: trainX, trainY: trainY, degree: degree,
                             fittedValues: fitted, sigma: sigma, trace: trace,
                             weights: robust, selectedNeighborhoods: selected,
                             bandwidths: bandwidths, candidateNeighborhoods: candidates)
    }

    /// Predict at `x`: select (same unweighted recipe as training), then fit
    /// with the final robust weights.
    public func predict(_ x: [Double]) -> Double {
        guard x.count == trainX[0].count else { return .nan }
        let q = AdaptiveLoess.basisSize(degree: degree, dimensions: x.count)
        let search = NeighborSearch(trainX: trainX, forBatchUse: false)
        guard let s = AdaptiveLoess.select(search: search, trainY: trainY, degree: degree,
                                           parameters: q, at: x,
                                           candidates: candidateNeighborhoods) else { return .nan }
        guard let e = AdaptiveLoess.evaluate(search: search, trainY: trainY, degree: degree,
                                             at: x, neighborhood: s.neighborhood,
                                             robust: weights, track: nil) else { return .nan }
        return e.value
    }

    /// Approximate standard error over the selected neighborhood.
    public func standardError(at x: [Double]) -> Double? {
        guard x.count == trainX[0].count else { return nil }
        let q = AdaptiveLoess.basisSize(degree: degree, dimensions: x.count)
        let search = NeighborSearch(trainX: trainX, forBatchUse: false)
        guard let s = AdaptiveLoess.select(search: search, trainY: trainY, degree: degree,
                                           parameters: q, at: x,
                                           candidates: candidateNeighborhoods) else { return nil }
        return Loess.kernelStandardError(search: search, sigma: sigma, degree: degree,
                                          at: x, neighborhood: s.neighborhood)
    }

    /// Predictions over many queries (one shared neighbor index — much
    /// cheaper than looping `predict(_:)`).
    public func predict(_ xs: [[Double]]) -> [Double] {
        let search = NeighborSearch(trainX: trainX)
        let trainY = trainY
        let degree = degree
        let weights = weights
        let candidates = candidateNeighborhoods
        return xs.map { x in
            guard x.count == trainX[0].count else { return .nan }
            let q = AdaptiveLoess.basisSize(degree: degree, dimensions: x.count)
            guard let s = AdaptiveLoess.select(search: search, trainY: trainY, degree: degree,
                                               parameters: q, at: x,
                                               candidates: candidates) else { return .nan }
            guard let e = AdaptiveLoess.evaluate(search: search, trainY: trainY, degree: degree,
                                                 at: x, neighborhood: s.neighborhood,
                                                 robust: weights, track: nil) else { return .nan }
            return e.value
        }
    }

    /// Concurrent batch predictions (identical to `predict(_:)`).
    public func predictConcurrently(_ xs: [[Double]]) async -> [Double] {
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let trainY = trainY
        let degree = degree
        let weights = weights
        let candidates = candidateNeighborhoods
        return await concurrentMap(over: xs.count) { i in
            let x = xs[i]
            guard x.count == trainX[0].count else { return .nan }
            let q = AdaptiveLoess.basisSize(degree: degree, dimensions: x.count)
            guard let s = AdaptiveLoess.select(search: search, trainY: trainY, degree: degree,
                                               parameters: q, at: x,
                                               candidates: candidates) else { return .nan }
            guard let e = AdaptiveLoess.evaluate(search: search, trainY: trainY, degree: degree,
                                                 at: x, neighborhood: s.neighborhood,
                                                 robust: weights, track: nil) else { return .nan }
            return e.value
        }
    }

    /// Standard errors over many queries (one shared neighbor index).
    public func standardErrors(at xs: [[Double]]) -> [Double?] {
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let trainY = trainY
        let degree = degree
        let sigma = sigma
        let candidates = candidateNeighborhoods
        return xs.map { x in
            guard x.count == trainX[0].count else { return nil }
            let q = AdaptiveLoess.basisSize(degree: degree, dimensions: x.count)
            guard let s = AdaptiveLoess.select(search: search, trainY: trainY, degree: degree,
                                               parameters: q, at: x,
                                               candidates: candidates) else { return nil }
            return Loess.kernelStandardError(search: search, sigma: sigma, degree: degree,
                                              at: x, neighborhood: s.neighborhood)
        }
    }

    /// Concurrent batch standard errors (identical to `standardErrors(at:)`).
    public func standardErrorsConcurrently(at xs: [[Double]]) async -> [Double?] {
        let search = NeighborSearch(trainX: trainX)
        let trainX = trainX
        let trainY = trainY
        let degree = degree
        let sigma = sigma
        let candidates = candidateNeighborhoods
        return await concurrentMap(over: xs.count) { i in
            let x = xs[i]
            guard x.count == trainX[0].count else { return nil }
            let q = AdaptiveLoess.basisSize(degree: degree, dimensions: x.count)
            guard let s = AdaptiveLoess.select(search: search, trainY: trainY, degree: degree,
                                               parameters: q, at: x,
                                               candidates: candidates) else { return nil }
            return Loess.kernelStandardError(search: search, sigma: sigma, degree: degree,
                                              at: x, neighborhood: s.neighborhood)
        }
    }
}
