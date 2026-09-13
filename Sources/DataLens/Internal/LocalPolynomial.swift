import Foundation

/// Shared weighted-local-polynomial engine behind `Loess` and `AdaptiveLoess`.
///
/// Pure numerics over caller-supplied basis rows: sqrt-weight scaling, QR
/// least squares, own-leverage via the normal equations. Returns `nil` when
/// the weighted design is rank-deficient (callers fall back to a bounded
/// local mean). Extracted so both smoothers share one code path — same
/// operations in the same order as the original `Loess.localFit` body.
enum LocalPolynomial {
    /// Fitted coefficients (in the caller's centered basis; `[0]` is the
    /// value at the center) plus own-leverage (0 when `track` is nil).
    struct Fit: Sendable {
        let coefficients: [Double]
        let leverage: Double
    }

    /// - Parameters:
    ///   - rows: unscaled basis rows (caller centers them at the fit point).
    ///   - values: responses aligned with `rows`.
    ///   - weights: non-negative weights aligned with `rows`.
    ///   - track: position in the arrays whose leverage to compute, or nil.
    static func fitWeighted(rows: [[Double]], values: [Double], weights: [Double],
                            track: Int?) -> Fit? {
        guard !rows.isEmpty, rows.count == values.count, rows.count == weights.count else { return nil }
        var X: [[Double]] = []
        X.reserveCapacity(rows.count)
        var y: [Double] = []
        y.reserveCapacity(rows.count)
        for (row, (value, weight)) in zip(rows, zip(values, weights)) {
            let s = sqrt(weight)
            X.append(row.map { $0 * s })
            y.append(value * s)
        }
        guard let beta = LinAlg.leastSquares(design: X, response: y) else { return nil }
        // Leverage l_t(x) = e₁ᵀ(XᵀWX)⁻¹·(XᵀW e_t): solve normal equations once.
        var leverage = 0.0
        if let pos = track, rows.indices.contains(pos) {
            let q = beta.count
            var XtX = [[Double]](repeating: [Double](repeating: 0, count: q), count: q)
            var Xtw = [Double](repeating: 0, count: q)
            for (r, (row, weight)) in zip(rows, weights).enumerated() {
                for a in 0..<q {
                    Xtw[a] += row[a] * (r == pos ? weight : 0)
                    for b in 0..<q { XtX[a][b] += row[a] * weight * row[b] }
                }
            }
            if let col = Regression.solve(XtX, Xtw) {
                leverage = col[0]
            }
        }
        return Fit(coefficients: beta, leverage: leverage)
    }
}
