import Foundation

/// Vendored subset of Numerical-Statistics' `Regression` (byte-identical
/// numerics, demoted to internal): square-system solver only.
///
/// Least-squares fits go through ``LinAlg`` — never form XᵀX for fitting.
/// This solver is for Newton-type square systems and, in this package, for
/// the LOESS leverage/SE weights.
enum Regression {
    /// Solve a square linear system with partial-pivot Gaussian elimination.
    static func solve(_ A: [[Double]], _ b: [Double]) -> [Double]? {
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
