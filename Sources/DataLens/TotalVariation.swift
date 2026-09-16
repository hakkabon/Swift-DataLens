import Foundation

/// Total-variation denoising (1-D fused lasso): piecewise-constant fits
/// minimizing `½‖x−y‖² + λ‖Dx‖₁` over the x-ordered sequence, via ADMM
/// with a banded direct solve per round.
///
/// Deliberately un-clever: the15-line Condat direct algorithm would be
/// faster, but ADMM is correct by construction ( textbook Boyd splits:
/// quadratic x-step through `BandedMatrix`, soft-threshold z-step,
/// dual ascent) and every fit is verified against the KKT conditions in
/// tests — a wrong direct implementation would fail loudly there, but a
/// right-by-construction one cannot be wrong in the first place.
/// Iterative per the house rules (residual stopping test, iteration cap,
/// nil — never unconverged values — past the cap).
///
/// Uncertainty is homoskedastic by design: the piecewise-constant fit
/// has no meaningful pointwise leverage, so every standard error is the
/// residual scale σ (documented approximation, not a gap). Trace for GCV
/// is the segment count (Tibshirani–Taylor), estimated under a relative
/// jump tolerance. Sequence semantics mirror `WhittakerEilers`: rows are
/// ordered by their single predictor, spacing ignored, single predictor
/// only (v1 scope), missing rows drop with `keptIndices`.
public struct TotalVariation: Sendable {
    public let trainX: [[Double]]
    public let trainY: [Double]
    public let lambda: Double
    /// Fitted values at the training points (survivor order).
    public let fittedValues: [Double]
    /// Residual scale σ̂ = √(RSS/max(n−tr,1)).
    public let sigma: Double
    /// Estimated degrees of freedom (fused-segment count, see above).
    public let trace: Double
    /// Residual scale at every point (homoskedastic approximation).
    public let standardErrors: [Double]
    /// Survivor positions for `droppingMissing` (identity otherwise).
    public let keptIndices: [Int]

    private let sortedX: [Double]
    private let sortedFitted: [Double]

    /// ADMM penalty parameter (fixed: the splits converge for any ρ > 0;
    /// tuning it buys little and costs an API knob).
    static let rho = 1.0
    /// Convergence tolerance, scaled by the response RMS at fit time.
    static let tolerance = 1e-10
    /// Iteration cap: well-posed fits converge in the hundreds; past the
    /// cap the fit returns nil instead of unconverged values.
    static let maxIterations = 20000
    /// Relative jump tolerance for segment counting (trace only).
    static let segmentTolerance = 1e-6

    init(trainX: [[Double]], trainY: [Double], lambda: Double,
         fittedValues: [Double], sigma: Double, trace: Double,
         standardErrors: [Double], keptIndices: [Int],
         sortedX: [Double], sortedFitted: [Double]) {
        self.trainX = trainX
        self.trainY = trainY
        self.lambda = lambda
        self.fittedValues = fittedValues
        self.sigma = sigma
        self.trace = trace
        self.standardErrors = standardErrors
        self.keptIndices = keptIndices
        self.sortedX = sortedX
        self.sortedFitted = sortedFitted
    }

    /// Fit by ADMM. `lambda = 0` reproduces the responses (up to the
    /// convergence tolerance — iterative, not bit-exact).
    ///
    /// With `droppingMissing`, rows with non-finite coordinates or responses
    /// are dropped first (`keptIndices` records the survivors); otherwise
    /// such rows fail validation and the fit is nil. Non-single-predictor
    /// input is v1-out-of-scope and returns nil.
    public static func fit(trainX: [[Double]], trainY: [Double],
                           lambda: Double,
                           droppingMissing: Bool = false) -> TotalVariation? {
        guard trainX.count == trainY.count else { return nil }
        let (trainX, trainY, keptIndices): ([[Double]], [Double], [Int]) = droppingMissing
            ? MissingData.dropping(trainX: trainX, trainY: trainY)
            : (trainX, trainY, Array(trainX.indices))
        guard !trainX.isEmpty,
              trainX.allSatisfy({ $0.count == 1 }),
              lambda.isFinite, lambda >= 0,
              trainX.flatMap({ $0 }).allSatisfy({ $0.isFinite }),
              trainY.allSatisfy({ $0.isFinite }) else { return nil }
        guard let solved = solveADMM(xs: trainX.map({ $0[0] }), ys: trainY, lambda: lambda) else {
            return nil
        }
        return assemble(trainX: trainX, trainY: trainY, lambda: lambda,
                        keptIndices: keptIndices, solved: solved)
    }

    /// Concurrent fit: identical to `fit` (ADMM rounds are sequential, so
    /// there is no parallel structure to exploit; this exists for API
    /// symmetry and checks cancellation around the synchronous work).
    public static func fitConcurrently(trainX: [[Double]], trainY: [Double],
                                       lambda: Double,
                                       droppingMissing: Bool = false) async throws -> TotalVariation? {
        try Task.checkCancellation()
        let result = fit(trainX: trainX, trainY: trainY, lambda: lambda, droppingMissing: droppingMissing)
        try Task.checkCancellation()
        return result
    }

    /// Solved system in x-sorted order, plus the permutation back.
    private struct Solved: Sendable {
        var fitted: [Double]
        var order: [Int]  // solve position → survivor index
    }

    /// Stable x-order (decorated: Swift's sort is not stable, and
    /// duplicate x must keep file order).
    static func sortOrder(xs: [Double]) -> [Int] {
        xs.indices.sorted { xs[$0] < xs[$1] || (xs[$0] == xs[$1] && $0 < $1) }
    }

    /// Forward differences over the sorted sequence.
    static func differences(_ v: [Double]) -> [Double] {
        guard v.count >= 2 else { return [] }
        return (0..<(v.count - 1)).map { v[$0 + 1] - v[$0] }
    }

    /// Adjoint differences: `(Dᵀv)[0] = −v[0]`, `(Dᵀv)[i] = v[i−1] − v[i]`,
    /// `(Dᵀv)[n−1] = v[n−2]`.
    static func adjointDifferences(_ v: [Double], n: Int) -> [Double] {
        guard n >= 1 else { return [] }
        guard n >= 2 else { return [0] }
        var out = [Double](repeating: 0, count: n)
        out[0] = -v[0]
        for i in 1..<(n - 1) { out[i] = v[i - 1] - v[i] }
        out[n - 1] = v[n - 2]
        return out
    }

    static func shrink(_ t: Double, _ kappa: Double) -> Double {
        if t > kappa { return t - kappa }
        if t < -kappa { return t + kappa }
        return 0
    }

    private static func solveADMM(xs: [Double], ys: [Double], lambda: Double) -> Solved? {
        let n = xs.count
        let permutation = sortOrder(xs: xs)
        let sy = permutation.map { ys[$0] }
        // M = I + ρDᵀD is tridiagonal: diag 1+ρ·(1|2), off −ρ.
        let matrix = BandedMatrix(n: n, bandwidth: 1) { i, j in
            if i == j {
                let degree = (i > 0 ? 1 : 0) + (i < n - 1 ? 1 : 0)
                return 1 + rho * Double(degree)
            }
            return -rho
        }
        guard let factor = matrix.cholesky() else { return nil }
        let rms = sqrt(sy.reduce(0.0) { $0 + $1 * $1 } / Double(n))
        let eps = tolerance * max(1.0, rms)
        var x = sy
        var z = differences(sy)
        var u = [Double](repeating: 0, count: max(0, n - 1))
        var zPrev = z
        for _ in 0..<maxIterations {
            // x-step: (I + ρDᵀD)x = y + ρDᵀ(z − u).
            let zu = zip(z, u).map { $0 - $1 }
            let rhs = zip(sy, adjointDifferences(zu, n: n)).map { $0 + rho * $1 }
            x = factor.solve(rhs)
            // z-step: soft-threshold the shifted differences.
            let dx = differences(x)
            zPrev = z
            z = zip(dx, u).map { shrink($0 + $1, lambda / rho) }
            // Dual ascent.
            u = zip(zip(dx, z).map { $0 - $1 }, u).map { $0 + $1 }
            // Stopping: primal and dual residuals small.
            let primal = sqrt(zip(dx, z).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) })
            let dualStep = zip(z, zPrev).map { $0 - $1 }
            let dual = rho * sqrt(adjointDifferences(dualStep, n: n).reduce(0.0) { $0 + $1 * $1 })
            if primal <= eps, dual <= eps {
                return Solved(fitted: x, order: permutation)
            }
        }
        return nil
    }

    /// Map a solved system back to survivor order and assemble the fit.
    private static func assemble(trainX: [[Double]], trainY: [Double], lambda: Double,
                                 keptIndices: [Int], solved: Solved) -> TotalVariation {
        let n = trainX.count
        var fitted = [Double](repeating: 0, count: n)
        var sortedX = [Double](repeating: 0, count: n)
        for (p, s) in solved.order.enumerated() {
            fitted[s] = solved.fitted[p]
            sortedX[p] = trainX[solved.order[p]][0]
        }
        // Segment-count trace (Tibshirani–Taylor): jumps above a relative
        // tolerance start new segments. An estimate — documented in the
        // type notes — sufficient for GCV comparison, never exact.
        let yRange = (trainY.max() ?? 0) - (trainY.min() ?? 0)
        let jumpTol = segmentTolerance * max(1.0, yRange)
        var segments = 1
        if n >= 2 {
            for p in 1..<n where abs(solved.fitted[p] - solved.fitted[p - 1]) > jumpTol {
                segments += 1
            }
        }
        let trace = Double(segments)
        let rss = zip(trainY, fitted).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let sigma = sqrt(rss / max(Double(n) - trace, 1))
        return TotalVariation(
            trainX: trainX, trainY: trainY, lambda: lambda,
            fittedValues: fitted, sigma: sigma, trace: trace,
            standardErrors: [Double](repeating: sigma, count: n),
            keptIndices: keptIndices,
            sortedX: sortedX, sortedFitted: solved.fitted
        )
    }

    /// GCV score over candidate lambdas (uses the segment-count trace).
    public static func selectLambda(trainX: [[Double]], trainY: [Double],
                                    lambdas: [Double],
                                    droppingMissing: Bool = false) -> (lambda: Double, fit: TotalVariation)? {
        var best: (lambda: Double, fit: TotalVariation)?
        var bestScore = Double.infinity
        for lambda in lambdas {
            guard let fit = TotalVariation.fit(trainX: trainX, trainY: trainY, lambda: lambda,
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

    /// Nearest sorted-training index to `x` (ties → lower index).
    private func nearestIndex(_ x: Double) -> Int? {
        guard !sortedX.isEmpty, x.isFinite else { return nil }
        var lo = 0
        var hi = sortedX.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if sortedX[mid] < x {
                lo = mid + 1
            } else {
                hi = mid
            }
        }
        if lo == 0 { return 0 }
        if lo >= sortedX.count { return sortedX.count - 1 }
        return (x - sortedX[lo - 1]) <= (sortedX[lo] - x) ? lo - 1 : lo
    }

    /// Predict by nearest sorted training fitted value (ties → lower
    /// index, deterministically). `.nan` outside the hull or on width
    /// mismatch; `.nearest` returns the endpoint value instead.
    public func predict(_ x: [Double],
                        extrapolation: ExtrapolationPolicy = .polynomial) -> Double {
        guard x.count == trainX[0].count, x[0].isFinite,
              let first = sortedX.first, let last = sortedX.last else { return .nan }
        if x[0] < first || x[0] > last {
            switch extrapolation {
            case .polynomial, .unavailable:
                return .nan
            case .nearest:
                return sortedFitted[x[0] < first ? 0 : sortedFitted.count - 1]
            }
        }
        guard let j = nearestIndex(x[0]) else { return .nan }
        return sortedFitted[j]
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

    /// Gradient ∇ŷ(x): zero almost everywhere (piecewise-constant fit),
    /// nil outside the hull and on width mismatch.
    public func gradient(at x: [Double]) -> [Double]? {
        guard x.count == trainX[0].count, !x.isEmpty, x[0].isFinite,
              let first = sortedX.first, let last = sortedX.last,
              x[0] >= first, x[0] <= last else { return nil }
        return [Double](repeating: 0, count: x.count)
    }

    /// Gradients over many queries.
    public func gradients(at xs: [[Double]]) -> [[Double]?] {
        xs.map { gradient(at: $0) }
    }

    /// Concurrent batch gradients (identical to `gradients(at:)`).
    public func gradientsConcurrently(at xs: [[Double]]) async throws -> [[Double]?] {
        try await concurrentMap(over: xs.count) { i in gradient(at: xs[i]) }
    }

    /// Standard error at `x` (homoskedastic σ; nil outside the hull or on
    /// width mismatch, `.nearest` returns σ at endpoints).
    public func standardError(at x: [Double],
                              extrapolation: ExtrapolationPolicy = .polynomial) -> Double? {
        guard x.count == trainX[0].count, x[0].isFinite,
              let first = sortedX.first, let last = sortedX.last else { return nil }
        if x[0] < first || x[0] > last {
            switch extrapolation {
            case .polynomial, .unavailable:
                return nil
            case .nearest:
                return sigma
            }
        }
        guard nearestIndex(x[0]) != nil else { return nil }
        return sigma
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
