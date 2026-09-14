import Foundation
#if canImport(NumericCoreAccelerate)
import NumericCore
import NumericCoreAccelerate
#endif

/// Square-system solver: LAPACK (`AccelerateBackend.solve`, QR-based) on
/// Apple platforms, vendored Gaussian elimination below elsewhere (Linux) —
/// same signature, same nil-on-singularity contract.
///
/// Least-squares fits go through ``LinAlg`` — never form XᵀX for fitting.
/// This solver is for Newton-type square systems and, in this package, for
/// the LOESS leverage/SE weights. The SPD fast path (`solveSPD`) is kept
/// separate for local likelihood — the frozen LOESS calls stay on `solve`.
enum Regression {
    /// Solve a square linear system (QR-based on Apple, partial-pivot
    /// Gaussian elimination in the Linux fallback). Returns `nil` on
    /// singularity (near-zero pivot / rank-deficient R diagonal).
    static func solve(_ A: [[Double]], _ b: [Double]) -> [Double]? {
        #if canImport(NumericCoreAccelerate)
        do {
            let a = try Matrix<Double>(rows: A)
            let rhs = Vector(b)
            guard let v = try? AccelerateBackend.solve(a, rhs) else { return nil }
            return v.storage
        } catch {
            return nil
        }
        #else
        return eliminate(A, b)
        #endif
    }

    /// SPD fast path for local-likelihood normal equations (XᵀWX, symmetric
    /// by construction): Cholesky (dpotrf/dpotrs, ~2x fewer flops than QR)
    /// on Apple, scalar Cholesky below elsewhere (correct, same verdicts).
    /// Returns `nil` iff A is not positive-definite.
    ///
    /// - Warning: symmetry is *not* checked — LAPACK reads one triangle
    ///   only (and the fallback mirrors that with the upper triangle).
    ///   Call only with symmetric-by-construction matrices; otherwise
    ///   use `solve(_:_:)`.
    static func solveSPD(_ A: [[Double]], _ b: [Double]) -> [Double]? {
        #if canImport(NumericCoreAccelerate)
        do {
            let a = try Matrix<Double>(rows: A)
            let rhs = Vector(b)
            guard let v = try? AccelerateBackend.solveSPD(a, rhs) else { return nil }
            return v.storage
        } catch {
            return nil
        }
        #else
        return cholesky(A, b)
        #endif
    }

    /// Scalar Cholesky solve (Linux fallback for `solveSPD`): factor
    /// A = UᵀU from the upper triangle — mirroring LAPACK's `uplo = "U"` —
    /// then forward/back substitution. Returns `nil` on empty/mismatched
    /// input or a non-positive pivot, the same verdict `dpotrf` renders
    /// (definite/not-definite only, not conditioning quality).
    static func cholesky(_ A: [[Double]], _ b: [Double]) -> [Double]? {
        let n = b.count
        guard A.count == n, A.allSatisfy({ $0.count == n }) else { return nil }
        var U = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for j in i..<n {
                var s = A[i][j]
                for k in 0..<i { s -= U[k][i] * U[k][j] }
                if i == j {
                    guard s > 0 else { return nil }
                    U[i][i] = sqrt(s)
                } else {
                    U[i][j] = s / U[i][i]
                }
            }
        }
        // Solve Uᵀy = b (forward), then Ux = y (backward).
        var y = b
        for i in 0..<n {
            var s = y[i]
            for k in 0..<i { s -= U[k][i] * y[k] }
            y[i] = s / U[i][i]
        }
        var x = y
        for i in stride(from: n - 1, through: 0, by: -1) {
            var s = x[i]
            for k in i + 1..<n { s -= U[i][k] * x[k] }
            x[i] = s / U[i][i]
        }
        return x
    }

    /// Partial-pivot Gaussian elimination (Linux fallback for `solve`).
    private static func eliminate(_ A: [[Double]], _ b: [Double]) -> [Double]? {
        let n = b.count
        guard A.count == n, A.allSatisfy({ $0.count == n }) else { return nil }
        var M = A, x = b
        for col in 0..<n {
            var pivot = col
            for row in col..<n where abs(M[row][col]) > abs(M[pivot][col]) { pivot = row }
            guard abs(M[pivot][col]) > 1e-12 else { return nil }
            M.swapAt(col, pivot); x.swapAt(col, pivot)
            for row in col + 1..<n {
                let f = M[row][col] / M[col][col]
                for k in col..<n { M[row][k] -= f * M[col][k] }
                x[row] -= f * x[col]
            }
        }
        var sol = [Double](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var s = x[i]
            for j in i + 1..<n { s -= M[i][j] * sol[j] }
            sol[i] = s / M[i][i]
        }
        return sol
    }
}
