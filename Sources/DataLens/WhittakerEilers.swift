import Foundation

/// Whittaker–Eilers smoothing (Eilers 2003; Hodrick–Prescott is order 2):
/// penalized least squares over a sequence, `ŷ = (I + λDᵀD)⁻¹y` with `D`
/// the order-`d` difference matrix, solved banded in O(n·d²) through the
/// internal `BandedMatrix` (Cholesky for fits, Takahashi for the trace
/// and standard errors — the solve-only `LinAlg` seam cannot provide
/// those, hence the local band solver).
///
/// Sequence semantics, stated plainly: rows are ordered by their single
/// predictor and the penalty runs over positions, so `x` only orders —
/// spacing is ignored. For near-uniform sampled series (the method's
/// home ground) this is exact; for wild spacing, bin first. Single
/// predictor only (v1 scope). Missing rows drop with `keptIndices`,
/// mirroring the other smoothers (in-system weight-0 interpolation is a
/// documented follow-up, not this change).
///
/// Fit-time cost is O(n²·b²): one banded solve for the fit plus one per
/// row for exact standard errors — comfortable to ~10k rows, documented
/// rather than capped.
public struct WhittakerEilers: Sendable {
    public let trainX: [[Double]]
    public let trainY: [Double]
    public let lambda: Double
    public let order: Int
    /// Fitted values at the training points (survivor order).
    public let fittedValues: [Double]
    /// Residual scale σ̂ = √(RSS/max(n−tr,1)).
    public let sigma: Double
    /// Smoother trace (effective degrees of freedom, via Takahashi).
    public let trace: Double
    /// Pointwise standard errors at the training points (exact:
    /// σ·‖rowᵢ(S)‖ over full solved rows, survivor order).
    public let standardErrors: [Double]
    /// Survivor positions for `droppingMissing` (identity otherwise).
    public let keptIndices: [Int]

    private let sortedX: [Double]
    private let sortedFitted: [Double]
    private let sortedSE: [Double]

    init(trainX: [[Double]], trainY: [Double], lambda: Double, order: Int,
         fittedValues: [Double], sigma: Double, trace: Double,
         standardErrors: [Double], keptIndices: [Int],
         sortedX: [Double], sortedFitted: [Double], sortedSE: [Double]) {
        self.trainX = trainX
        self.trainY = trainY
        self.lambda = lambda
        self.order = order
        self.fittedValues = fittedValues
        self.sigma = sigma
        self.trace = trace
        self.standardErrors = standardErrors
        self.keptIndices = keptIndices
        self.sortedX = sortedX
        self.sortedFitted = sortedFitted
        self.sortedSE = sortedSE
    }

    /// Order-`d` difference coefficients: alternating binomial signs.
    static func differenceCoefficients(order: Int) -> [Double] {
        var coeffs = [1.0]
        for _ in 0..<order {
            var next = [Double](repeating: 0, count: coeffs.count + 1)
            for (i, c) in coeffs.enumerated() {
                next[i] += c
                next[i + 1] -= c
            }
            coeffs = next
        }
        return coeffs
    }

    /// Fit by penalized least squares. `lambda = 0` interpolates
    /// (fitted == responses, bit-identically); larger λ smooths harder.
    ///
    /// With `droppingMissing`, rows with non-finite coordinates or responses
    /// are dropped first (`keptIndices` records the survivors); otherwise
    /// such rows fail validation and the fit is nil. Non-single-predictor
    /// input is v1-out-of-scope and returns nil.
    public static func fit(trainX: [[Double]], trainY: [Double],
                           lambda: Double, order: Int = 2,
                           droppingMissing: Bool = false) -> WhittakerEilers? {
        guard trainX.count == trainY.count else { return nil }
        let (trainX, trainY, keptIndices): ([[Double]], [Double], [Int]) = droppingMissing
            ? MissingData.dropping(trainX: trainX, trainY: trainY)
            : (trainX, trainY, Array(trainX.indices))
        guard !trainX.isEmpty,
              trainX.allSatisfy({ $0.count == 1 }),
              (1...3).contains(order),
              lambda.isFinite, lambda >= 0,
              trainX.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              trainY.allSatisfy({ $0.isFinite }) else { return nil }
        guard let solved = solveSystem(xs: trainX.map({ $0[0] }), ys: trainY,
                                       lambda: lambda, order: order) else { return nil }
        return assemble(trainX: trainX, trainY: trainY, lambda: lambda, order: order,
                        keptIndices: keptIndices, solved: solved)
    }

    /// Concurrent fit: identical to `fit` (the `n` standard-error solves
    /// run in parallel; assembly stays sequential). Bit-identical to `fit`.
    public static func fitConcurrently(trainX: [[Double]], trainY: [Double],
                                       lambda: Double, order: Int = 2,
                                       droppingMissing: Bool = false) async throws -> WhittakerEilers? {
        guard trainX.count == trainY.count else { return nil }
        let (trainX, trainY, keptIndices): ([[Double]], [Double], [Int]) = droppingMissing
            ? MissingData.dropping(trainX: trainX, trainY: trainY)
            : (trainX, trainY, Array(trainX.indices))
        guard !trainX.isEmpty,
              trainX.allSatisfy({ $0.count == 1 }),
              (1...3).contains(order),
              lambda.isFinite, lambda >= 0,
              trainX.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              trainY.allSatisfy({ $0.isFinite }) else { return nil }
        guard let solved = try await solveSystemConcurrently(xs: trainX.map({ $0[0] }), ys: trainY,
                                                             lambda: lambda, order: order) else { return nil }
        return assemble(trainX: trainX, trainY: trainY, lambda: lambda, order: order,
                        keptIndices: keptIndices, solved: solved)
    }

    /// Solved system: fitted values, trace, residual scale, and per-point
    /// SEs, all in the (x-sorted) solve order, plus the sort permutation
    /// back to survivor order.
    private struct Solved: Sendable {
        var fitted: [Double]
        var trace: Double
        var sigma: Double
        var se: [Double]
        var order: [Int]  // solve position → survivor index
    }

    /// Stable x-order (decorated with survivor indices: Swift's sort is
    /// not stable, and duplicate x must keep file order).
    static func sortOrder(xs: [Double]) -> [Int] {
        xs.indices.sorted { xs[$0] < xs[$1] || (xs[$0] == xs[$1] && $0 < $1) }
    }

    private static func systemMatrix(n: Int, lambda: Double, order: Int) -> BandedMatrix? {
        let bandwidth = 2 * order
        let coeffs = differenceCoefficients(order: order)
        let matrix = BandedMatrix(n: n, bandwidth: bandwidth) { i, j in
            var v = (i == j) ? 1.0 : 0.0
            if lambda > 0, n > order {
                // (DᵀD)[i][j] over rows r with r ≤ i,j ≤ r+order, i.e.
                // r ∈ [i−order, j] (valid as i ≥ j here).
                let rLo = max(0, i - order)
                let rHi = min(j, n - 1 - order)
                if rLo <= rHi {
                    for r in rLo...rHi {
                        v += lambda * coeffs[i - r] * coeffs[j - r]
                    }
                }
            }
            return v
        }
        return matrix
    }

    private static func solveSystem(xs: [Double], ys: [Double], lambda: Double, order: Int) -> Solved? {
        let n = xs.count
        let permutation = sortOrder(xs: xs)
        let sy = permutation.map { ys[$0] }
        guard let matrix = systemMatrix(n: n, lambda: lambda, order: order),
              let factor = matrix.cholesky() else { return nil }
        let fitted = factor.solve(sy)
        let inv = factor.inverseBand()
        var trace = 0.0
        for i in 0..<n { trace += inv[0][i] }
        let rss = zip(sy, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma = sqrt(rss / max(Double(n) - trace, 1))
        // Exact SEs: Var(ŷᵢ) = σ²·‖rowᵢ(S)‖² over full solved rows.
        var se = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let row = factor.solve((0..<n).map { $0 == i ? 1.0 : 0.0 })
            se[i] = sigma * sqrt(row.reduce(0.0) { $0 + $1 * $1 })
        }
        return Solved(fitted: fitted, trace: trace, sigma: sigma, se: se, order: permutation)
    }

    private static func solveSystemConcurrently(xs: [Double], ys: [Double],
                                                lambda: Double, order: Int) async throws -> Solved? {
        let n = xs.count
        let permutation = sortOrder(xs: xs)
        let sy = permutation.map { ys[$0] }
        guard let matrix = systemMatrix(n: n, lambda: lambda, order: order),
              let factor = matrix.cholesky() else { return nil }
        let fitted = factor.solve(sy)
        let inv = factor.inverseBand()
        var trace = 0.0
        for i in 0..<n { trace += inv[0][i] }
        let rss = zip(sy, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma = sqrt(rss / max(Double(n) - trace, 1))
        let rows: [[Double]] = try await concurrentMap(over: n) { i in
            factor.solve((0..<n).map { $0 == i ? 1.0 : 0.0 })
        }
        var se = [Double](repeating: 0, count: n)
        for (i, row) in rows.enumerated() {
            se[i] = sigma * sqrt(row.reduce(0.0) { $0 + $1 * $1 })
        }
        return Solved(fitted: fitted, trace: trace, sigma: sigma, se: se, order: permutation)
    }

    /// Map a solved system back to survivor order and assemble the fit.
    private static func assemble(trainX: [[Double]], trainY: [Double], lambda: Double, order: Int,
                                 keptIndices: [Int], solved: Solved) -> WhittakerEilers {
        let n = trainX.count
        var fitted = [Double](repeating: 0, count: n)
        var se = [Double](repeating: 0, count: n)
        var sortedX = [Double](repeating: 0, count: n)
        for (p, s) in solved.order.enumerated() {
            fitted[s] = solved.fitted[p]
            se[s] = solved.se[p]
            sortedX[p] = trainX[solved.order[p]][0]
        }
        return WhittakerEilers(
            trainX: trainX, trainY: trainY, lambda: lambda, order: order,
            fittedValues: fitted, sigma: solved.sigma,
            trace: solved.trace, standardErrors: se, keptIndices: keptIndices,
            sortedX: sortedX, sortedFitted: solved.fitted, sortedSE: solved.se
        )
    }

    /// GCV score over candidate lambdas (uses each fit's trace).
    public static func selectLambda(trainX: [[Double]], trainY: [Double],
                                    lambdas: [Double], order: Int = 2,
                                    droppingMissing: Bool = false) -> (lambda: Double, fit: WhittakerEilers)? {
        var best: (lambda: Double, fit: WhittakerEilers)?
        var bestScore = Double.infinity
        for lambda in lambdas {
            guard let fit = WhittakerEilers.fit(trainX: trainX, trainY: trainY, lambda: lambda,
                                                order: order,
                                                droppingMissing: droppingMissing) else { continue }
            // Score on the fit's own (possibly dropped) rows (see DECISIONS #20).
            let n = Double(fit.trainY.count)
            let rss = zip(fit.trainY, fit.fittedValues).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
            let denom = max(1 - fit.trace / n, 1e-6)
            let score = (rss / n) / (denom * denom)
            if score < bestScore { bestScore = score; best = (lambda, fit) }
        }
        return best
    }

    /// Bracketing segment for `x` in the sorted grid, or nil outside it.
    private func bracket(_ x: Double) -> (lo: Int, hi: Int)? {
        guard sortedX.count >= 2, x.isFinite,
              let first = sortedX.first, let last = sortedX.last,
              x >= first, x <= last else { return nil }
        var lo = 0
        var hi = sortedX.count - 1
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if sortedX[mid] <= x {
                lo = mid
            } else {
                hi = mid
            }
        }
        return (lo, hi)
    }

    /// Predict by linear interpolation on the fitted grid (`.nan`
    /// outside the hull or on width mismatch — interpolation only).
    /// Under `.nearest`, out-of-hull queries return the nearest endpoint
    /// value instead; `.unavailable` matches the default `.nan`.
    public func predict(_ x: [Double],
                        extrapolation: ExtrapolationPolicy = .polynomial) -> Double {
        guard x.count == trainX[0].count, x[0].isFinite else { return .nan }
        if let first = sortedX.first, let last = sortedX.last,
           (x[0] < first || x[0] > last)
        {
            switch extrapolation {
            case .polynomial, .unavailable:
                return .nan
            case .nearest:
                return sortedFitted[x[0] < first ? 0 : sortedFitted.count - 1]
            }
        }
        guard let (lo, hi) = bracket(x[0]) else { return .nan }
        let x0 = sortedX[lo]
        let x1 = sortedX[hi]
        guard x1 > x0 else { return sortedFitted[lo] }
        let t = (x[0] - x0) / (x1 - x0)
        return sortedFitted[lo] * (1 - t) + sortedFitted[hi] * t
    }

    /// Predictions over many queries.
    public func predict(_ xs: [[Double]],
                        extrapolation: ExtrapolationPolicy = .polynomial) -> [Double] {
        xs.map { predict($0, extrapolation: extrapolation) }
    }

    /// Concurrent batch predictions (identical to `predict(_:)`).
    public func predictConcurrently(_ xs: [[Double]],
                                    extrapolation: ExtrapolationPolicy = .polynomial) async throws -> [Double] {
        try await concurrentMap(over: xs.count) { i in predict(xs[i], extrapolation: extrapolation) }
    }

    /// Gradient ∇ŷ(x): slope of the bracketing interpolation segment.
    /// Nil outside the hull, on degenerate segments, and on width mismatch.
    public func gradient(at x: [Double]) -> [Double]? {
        guard x.count == trainX[0].count, !x.isEmpty, x[0].isFinite,
              let (lo, hi) = bracket(x[0]) else { return nil }
        let dx = sortedX[hi] - sortedX[lo]
        guard dx > 0, dx.isFinite else { return nil }
        var grad = [Double](repeating: 0, count: x.count)
        grad[0] = (sortedFitted[hi] - sortedFitted[lo]) / dx
        return grad
    }

    /// Gradients over many queries.
    public func gradients(at xs: [[Double]]) -> [[Double]?] {
        xs.map { gradient(at: $0) }
    }

    /// Concurrent batch gradients (identical to `gradients(at:)`).
    public func gradientsConcurrently(at xs: [[Double]]) async throws -> [[Double]?] {
        try await concurrentMap(over: xs.count) { i in gradient(at: xs[i]) }
    }

    /// Standard error at `x` by interpolation of the training SEs
    /// (nil outside the hull or on width mismatch; `.nearest` returns
    /// the nearest endpoint SE instead).
    public func standardError(at x: [Double],
                              extrapolation: ExtrapolationPolicy = .polynomial) -> Double? {
        guard x.count == trainX[0].count, x[0].isFinite else { return nil }
        if let first = sortedX.first, let last = sortedX.last,
           (x[0] < first || x[0] > last)
        {
            switch extrapolation {
            case .polynomial, .unavailable:
                return nil
            case .nearest:
                return sortedSE[x[0] < first ? 0 : sortedSE.count - 1]
            }
        }
        guard let (lo, hi) = bracket(x[0]) else { return nil }
        let x0 = sortedX[lo]
        let x1 = sortedX[hi]
        guard x1 > x0 else { return sortedSE[lo] }
        let t = (x[0] - x0) / (x1 - x0)
        return sortedSE[lo] * (1 - t) + sortedSE[hi] * t
    }

    /// Standard errors over many queries.
    public func standardErrors(at xs: [[Double]],
                               extrapolation: ExtrapolationPolicy = .polynomial) -> [Double?] {
        xs.map { standardError(at: $0, extrapolation: extrapolation) }
    }

    /// Concurrent batch standard errors (identical to `standardErrors(at:)`).
    public func standardErrorsConcurrently(at xs: [[Double]],
                                           extrapolation: ExtrapolationPolicy = .polynomial) async throws -> [Double?] {
        try await concurrentMap(over: xs.count) { i in standardError(at: xs[i], extrapolation: extrapolation) }
    }
}
