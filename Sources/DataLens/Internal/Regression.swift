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
    /// on Apple, elimination in the Linux fallback (correct, not faster).
    /// Returns `nil` iff A is not positive-definite (Apple) or hits a
    /// singular pivot (fallback).
    ///
    /// - Warning: symmetry is *not* checked — LAPACK reads one triangle
    ///   only. Call only with symmetric-by-construction matrices; otherwise
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
        return eliminate(A, b)
        #endif
    }

    /// Partial-pivot Gaussian elimination (Linux fallback for both solvers).
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
