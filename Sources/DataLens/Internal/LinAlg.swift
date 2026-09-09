import Foundation

/// Small dense linear algebra: Householder QR and least-squares solves.
///
/// Vendored subset of Numerical-Statistics' `LinAlg` (byte-identical
/// numerics, demoted to internal): the normal equations (XᵀX) square the
/// condition number, so all least-squares fits go through
/// `leastSquares(design:response:)`.
///
/// - Note: `qr(_:)` is currently unused by `Loess` but kept (and covered in
///   the sibling repo) so the seam stays complete; do not delete it as dead
///   code without a `docs/DECISIONS.md` entry.
enum LinAlg {
    /// One Householder reflector: offset k, vector v, squared norm.
    private struct Reflector: Sendable {
        let k: Int
        let v: [Double]
        let normSq: Double
    }

    /// Reduce A (m×n, m ≥ n) to upper-trapezoidal R, returning R plus the
    /// Householder reflectors with A = H₀···Hₙ₋₁·R.
    private static func reduce(_ A: [[Double]]) -> (R: [[Double]], refs: [Reflector])? {
        let m = A.count
        guard m > 0 else { return nil }
        let n = A[0].count
        guard n > 0, m >= n, A.allSatisfy({ $0.count == n }) else { return nil }
        var R = A
        var refs: [Reflector] = []
        for k in 0..<n {
            var norm = 0.0
            for i in k..<m { norm += R[i][k] * R[i][k] }
            norm = sqrt(norm)
            if norm == 0 {
                refs.append(Reflector(k: k, v: [], normSq: 0))
                continue
            }
            // v = x + sign(x₁)·‖x‖·e₁ (avoids cancellation).
            let sign: Double = R[k][k] >= 0 ? 1 : -1
            var v = [Double](repeating: 0, count: m - k)
            for i in k..<m { v[i - k] = R[i][k] }
            v[0] += sign * norm
            var vNormSq = 0.0
            for x in v { vNormSq += x * x }
            guard vNormSq > 0 else {
                refs.append(Reflector(k: k, v: [], normSq: 0))
                continue
            }
            // R ← H_k·R on rows k..., cols k....
            for j in k..<n {
                var dot = 0.0
                for i in k..<m { dot += v[i - k] * R[i][j] }
                let f = 2 * dot / vNormSq
                for i in k..<m { R[i][j] -= f * v[i - k] }
            }
            refs.append(Reflector(k: k, v: v, normSq: vNormSq))
        }
        return (R, refs)
    }

    /// Apply H_k to a full-length vector in place.
    private static func apply(_ ref: Reflector, to w: inout [Double]) {
        guard !ref.v.isEmpty else { return }
        var dot = 0.0
        for i in ref.k..<w.count { dot += ref.v[i - ref.k] * w[i] }
        let f = 2 * dot / ref.normSq
        for i in ref.k..<w.count { w[i] -= f * ref.v[i - ref.k] }
    }

    /// Thin Householder QR: A (m×n, m ≥ n) = Q·R with Q m×n orthonormal
    /// columns and R n×n upper triangular. Returns `nil` if A is rank
    /// deficient (a diagonal of R falls below tolerance).
    static func qr(_ A: [[Double]]) -> (Q: [[Double]], R: [[Double]])? {
        guard let (R, refs) = reduce(A) else { return nil }
        let m = A.count, n = A[0].count
        let scale = abs(R[0][0])
        let tol = 1e-12 * max(scale, 1)
        for i in 0..<n {
            guard abs(R[i][i]) > tol else { return nil }
        }
        // Thin Q: j-th column is H₀···Hₙ₋₁·eⱼ (apply innermost first).
        var Q = [[Double]](repeating: [Double](repeating: 0, count: n), count: m)
        for j in 0..<n {
            var e = [Double](repeating: 0, count: m)
            e[j] = 1
            for ref in refs.reversed() { apply(ref, to: &e) }
            for i in 0..<m { Q[i][j] = e[i] }
        }
        let thinR = (0..<n).map { i in (0..<n).map { j in j < i ? 0 : R[i][j] } }
        return (Q, thinR)
    }

    /// Least-squares solution min ‖y − Xβ‖ via thin Householder QR.
    /// Returns `nil` when X is rank deficient.
    static func leastSquares(design X: [[Double]], response y: [Double]) -> [Double]? {
        guard !X.isEmpty, X.count == y.count else { return nil }
        guard let (R, refs) = reduce(X) else { return nil }
        let m = X.count, n = X[0].count
        let scale = abs(R[0][0])
        let tol = 1e-12 * max(scale, 1)
        for i in 0..<n {
            guard abs(R[i][i]) > tol else { return nil }
        }
        // z = Qᵀy with Q = H₀···Hₙ₋₁: apply H₀ first (forward order).
        var z = y
        for ref in refs { apply(ref, to: &z) }
        // Back-substitute R·β = z[0..<n].
        var beta = [Double](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var s = z[i]
            for j in i + 1..<n { s -= R[i][j] * beta[j] }
            beta[i] = s / R[i][i]
        }
        return beta
    }
}
