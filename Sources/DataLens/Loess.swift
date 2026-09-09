import Foundation

/// Locally weighted scatterplot smoothing (Cleveland 1979 / Cleveland–Devlin
/// 1988, LOESS): local polynomial fits under tricube nearest-neighbor weights
/// with bisquare robustness iterations.
///
/// Self-contained by design: weighted least squares go through ``LinAlg`` and
/// neighbor search is brute force behind the internal
/// ``Loess/nearestIndices(to:count:)`` seam, where a kd-tree can drop in
/// later (as can a Swift-Numerics matrix backend) without touching call sites.
public enum LoessWeight: Sendable {
    /// Tricube (Cleveland's default): (1−u³)³ on [0,1).
    case tricube
    /// Bisquare (robustness step): (1−u²)² on [0,1).
    case bisquare

    public func callAsFunction(_ u: Double) -> Double {
        guard u >= 0, u < 1 else { return 0 }
        switch self {
        case .tricube:
            let t = 1 - u * u * u
            return t * t * t
        case .bisquare:
            let t = 1 - u * u
            return t * t
        }
    }
}

public struct Loess: Sendable {
    public let trainX: [[Double]]
    public let trainY: [Double]
    public let span: Double
    public let degree: Int
    /// Fitted values at the training points (final robustness iteration).
    public let fittedValues: [Double]
    /// Residual scale σ̂ = √(RSS/max(n−tr,1)).
    public let sigma: Double
    /// Smoother-matrix trace Σᵢ lᵢ(xᵢ) (effective degrees of freedom).
    public let trace: Double
    /// Final combined weights (locality × robustness) per training point.
    public let weights: [Double]

    private init(trainX: [[Double]], trainY: [Double], span: Double, degree: Int,
                 fittedValues: [Double], sigma: Double, trace: Double, weights: [Double]) {
        self.trainX = trainX
        self.trainY = trainY
        self.span = span
        self.degree = degree
        self.fittedValues = fittedValues
        self.sigma = sigma
        self.trace = trace
        self.weights = weights
    }

    /// Monomial basis at `x` centered on `c`: [1, (x−c), squares, cross terms].
    static func basis(_ x: [Double], center c: [Double], degree: Int) -> [Double] {
        var row = [1.0]
        let d = x.enumerated().map { $0.element - c[$0.offset] }
        if degree >= 1 { row.append(contentsOf: d) }
        if degree >= 2 {
            for i in 0..<d.count {
                for j in i..<d.count { row.append(d[i] * d[j]) }
            }
        }
        return row
    }

    /// Indices of the `count` nearest training rows to `x` (brute force;
    /// the seam a kd-tree will replace). Sorted nearest-first.
    static func nearestIndices(_ trainX: [[Double]], to x: [Double], count: Int) -> [Int] {
        let k = min(max(count, 1), trainX.count)
        let dists = trainX.map { row in
            sqrt(zip(row, x).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) })
        }
        return dists.indices.sorted { dists[$0] < dists[$1] }.prefix(k).map { $0 }
    }

    /// Fit by `robustIterations` bisquare reweighting rounds (4 matches R).
    public static func fit(trainX: [[Double]], trainY: [Double],
                           span: Double = 0.75, degree: Int = 2,
                           robustIterations: Int = 4) -> Loess? {
        guard !trainX.isEmpty, trainX.count == trainY.count,
              span > 0, span <= 1, (0...2).contains(degree),
              trainX.allSatisfy({ $0.count == trainX[0].count }),
              trainX.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              trainY.allSatisfy({ $0.isFinite }) else { return nil }
        let n = trainX.count
        let p = trainX[0].count
        let q = 1 + (degree >= 1 ? p : 0) + (degree >= 2 ? p * (p + 1) / 2 : 0)
        let k = min(n, max(Int(ceil(span * Double(n))), q + 1))
        guard k > 1 else { return nil }
        var robust = [Double](repeating: 1, count: n)
        var fitted = [Double](repeating: 0, count: n)
        // Scale floor: once the fit is (near-)exact, MAD → 0 and an unguarded
        // cutoff would downweight good points. Floor against the y-scale.
        let medY = Descriptive.median(trainY) ?? 0
        let yScale = max(Descriptive.median(trainY.map { abs($0 - medY) }) ?? 0, 1e-300)
        for _ in 0...robustIterations {
            for i in 0..<n {
                fitted[i] = Loess.localFit(trainX: trainX, trainY: trainY, degree: degree,
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
            let r = Loess.localFit(trainX: trainX, trainY: trainY, degree: degree,
                                   at: trainX[i], neighborhood: k, robust: robust, trackIndex: i)
            fitted[i] = r.value
            trace += r.leverage
        }
        let rss = zip(trainY, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma = sqrt(rss / max(Double(n) - trace, 1))
        return Loess(trainX: trainX, trainY: trainY, span: span, degree: degree,
                     fittedValues: fitted, sigma: sigma, trace: trace, weights: robust)
    }

    /// Max neighbor distance (bandwidth); 0 when all neighbors coincide.
    static func bandwidth(trainX: [[Double]], indices: [Int], at x: [Double]) -> Double {
        indices.map { j in
            sqrt(zip(trainX[j], x).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) })
        }.max() ?? 0
    }

    /// Local fit at `x`: value + own-leverage (for the smoother trace).
    static func localFit(trainX: [[Double]], trainY: [Double], degree: Int,
                         at x: [Double], neighborhood k: Int, robust: [Double],
                         trackIndex: Int? = nil) -> (value: Double, leverage: Double) {
        let nb = Loess.nearestIndices(trainX, to: x, count: k)
        let h = Loess.bandwidth(trainX: trainX, indices: nb, at: x)
        // Locality weights first (robustness applied after, so a degenerate
        // combined set can still fall back to a bounded local mean — never to
        // a single neighbor, which would lock in a masking 2-cycle).
        var lw: [(idx: Int, w: Double)] = []
        for j in nb {
            let d = sqrt(zip(trainX[j], x).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) })
            lw.append((j, h <= 0 ? 1 : LoessWeight.tricube(d / h)))
        }
        var w: [(idx: Int, w: Double)] = []
        for (j, l) in lw {
            let ww = l * robust[j]
            if ww > 0 { w.append((j, ww)) }
        }
        guard !w.isEmpty else {
            let total = lw.reduce(0.0) { $0 + $1.w }
            if total > 0 {
                let value = lw.reduce(0.0) { $0 + trainY[$1.idx] * $1.w } / total
                return (value, 0)
            }
            return (trainY[nb.first ?? 0], 0)
        }
        if degree == 0 || w.count == 1 {
            // Nadaraya–Watson fallback (also the degree-0 estimator).
            let total = w.reduce(0.0) { $0 + $1.w }
            let value = w.reduce(0.0) { $0 + $1.w * trainY[$1.idx] } / total
            var leverage = 0.0
            if let t = trackIndex, let own = w.first(where: { $0.idx == t }) {
                leverage = own.w / total
            }
            return (value, leverage)
        }
        var X: [[Double]] = [], y: [Double] = [], sw: [Double] = []
        for (j, ww) in w {
            let s = sqrt(ww)
            X.append(Loess.basis(trainX[j], center: x, degree: degree).map { $0 * s })
            y.append(trainY[j] * s)
            sw.append(ww)
        }
        guard let beta = LinAlg.leastSquares(design: X, response: y) else {
            let total = sw.reduce(0.0, +)
            let value = zip(w, sw).reduce(0.0) { $0 + trainY[$1.0.idx] * $1.1 } / total
            return (value, 0)
        }
        // Leverage l_t(x) = e₁ᵀ(XᵀWX)⁻¹·(XᵀW e_t): solve normal equations once.
        var leverage = 0.0
        if let t = trackIndex, let pos = w.firstIndex(where: { $0.idx == t }) {
            let q = beta.count
            var XtX = [[Double]](repeating: [Double](repeating: 0, count: q), count: q)
            var Xtw = [Double](repeating: 0, count: q)
            for (r, (j, ww)) in w.enumerated() {
                let row = Loess.basis(trainX[j], center: x, degree: degree)
                for a in 0..<q {
                    Xtw[a] += row[a] * (r == pos ? ww : 0)
                    for b in 0..<q { XtX[a][b] += row[a] * ww * row[b] }
                }
            }
            if let col = Regression.solve(XtX, Xtw) {
                leverage = col[0]
            }
        }
        return (beta[0], leverage)
    }

    /// Predict at `x` using the final robust weights (fallback cascade inside).
    public func predict(_ x: [Double]) -> Double {
        guard x.count == trainX[0].count else { return .nan }
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), 1))
        return Loess.localFit(trainX: trainX, trainY: trainY, degree: degree,
                              at: x, neighborhood: k, robust: weights).value
    }

    /// Approximate standard error σ̂·‖l(x)‖ with the equivalent kernel l(x).
    public func standardError(at x: [Double]) -> Double? {
        guard x.count == trainX[0].count else { return nil }
        let k = min(trainX.count, max(Int(ceil(span * Double(trainX.count))), 1))
        let nb = Loess.nearestIndices(trainX, to: x, count: k)
        let h = Loess.bandwidth(trainX: trainX, indices: nb, at: x)
        var rows: [[Double]] = [], ws: [Double] = []
        for j in nb {
            let d = sqrt(zip(trainX[j], x).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) })
            let lw = h <= 0 ? 1 : LoessWeight.tricube(d / h)
            if lw > 0 {
                rows.append(Loess.basis(trainX[j], center: x, degree: degree))
                ws.append(lw)
            }
        }
        guard !rows.isEmpty else { return nil }
        if degree == 0 {
            let total = ws.reduce(0.0, +)
            return sigma * sqrt(ws.reduce(0.0) { $0 + pow($1 / total, 2) })
        }
        let q = rows[0].count
        var XtX = [[Double]](repeating: [Double](repeating: 0, count: q), count: q)
        var Xtw: [[Double]] = []
        for (r, row) in rows.enumerated() {
            for a in 0..<q {
                for b in 0..<q { XtX[a][b] += row[a] * ws[r] * row[b] }
            }
            Xtw.append(row.map { $0 * ws[r] })
        }
        // l_i = row_i · (XᵀWX)⁻¹e₁: solve once, dot per row.
        var e1 = [Double](repeating: 0, count: q)
        e1[0] = 1
        guard let col = Regression.solve(XtX, e1) else { return nil }
        let normSq = rows.reduce(0.0) { acc, row in
            let l = zip(row, col).reduce(0.0) { $0 + $1.0 * $1.1 }
            return acc + l * l
        }
        return sigma * sqrt(normSq)
    }

    /// GCV score over candidate spans (uses each fit's trace).
    public static func selectSpan(trainX: [[Double]], trainY: [Double],
                                  spans: [Double], degree: Int = 2,
                                  robustIterations: Int = 4) -> (span: Double, fit: Loess)? {
        var best: (span: Double, fit: Loess)?
        var bestScore = Double.infinity
        for span in spans {
            guard let fit = Loess.fit(trainX: trainX, trainY: trainY, span: span,
                                      degree: degree, robustIterations: robustIterations) else { continue }
            let n = Double(trainX.count)
            let rss = zip(trainY, fit.fittedValues).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
            let denom = max(1 - fit.trace / n, 1e-6)
            let score = (rss / n) / (denom * denom)
            if score < bestScore { bestScore = score; best = (span, fit) }
        }
        return best
    }
}
