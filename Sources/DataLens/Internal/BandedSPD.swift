import Foundation

/// Symmetric band matrix, lower-band storage: `lower[d][i]` holds
/// `A[i][i−d]` for `0 ≤ d ≤ bandwidth` (zero where `i−d < 0`).
///
/// Exists because the Whittaker smoother needs more than solves: GCV
/// trace and pointwise standard errors need the band of the inverse,
/// which a solve-only seam cannot provide. Internal: the only consumer
/// is `WhittakerEilers`. Verified element-wise against dense references
/// (see `BandedSPDTests`).
struct BandedMatrix: Sendable {
    let n: Int
    let bandwidth: Int
    private var lower: [[Double]]

    /// Assemble from a fill closure over `(row, col)` with `col ≤ row`.
    init(n: Int, bandwidth: Int, fill: (Int, Int) -> Double) {
        precondition(n >= 0, "BandedMatrix requires non-negative size")
        precondition(bandwidth >= 0, "BandedMatrix requires non-negative bandwidth")
        self.n = n
        self.bandwidth = bandwidth
        var lower = [[Double]](repeating: [Double](repeating: 0, count: n), count: bandwidth + 1)
        for i in 0..<n {
            for d in 0...min(bandwidth, i) {
                lower[d][i] = fill(i, i - d)
            }
        }
        self.lower = lower
    }

    /// Read any entry (mirrored; zero outside the band).
    func value(row i: Int, col j: Int) -> Double {
        let d = abs(i - j)
        guard d <= bandwidth else { return 0 }
        return lower[d][max(i, j)]
    }

    /// Unit-lower + diagonal factorisation `A = LDLᵀ`, or nil on a
    /// non-positive pivot (mirroring `solveSPD`'s verdict contract —
    /// nil means "not SPD", never a trap).
    func cholesky() -> BandedCholesky? {
        var unit = [[Double]](repeating: [Double](repeating: 0, count: n), count: bandwidth + 1)
        var diag = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let j0 = max(0, i - bandwidth)
            for j in j0...i {
                var s = value(row: i, col: j)
                let k0 = max(j0, j - bandwidth)
                if k0 < j {
                    for k in k0..<j {
                        s -= unit[i - k][i] * unit[j - k][j] * diag[k]
                    }
                }
                if i == j {
                    guard s > 0 else { return nil }
                    diag[i] = s
                } else {
                    unit[i - j][i] = s / diag[j]
                }
            }
        }
        return BandedCholesky(n: n, bandwidth: bandwidth, unitLower: unit, diag: diag)
    }
}

/// Unit-lower + diagonal factor of a symmetric band matrix
/// (`A = LDLᵀ`; the unit diagonal is implicit).
struct BandedCholesky: Sendable {
    let n: Int
    let bandwidth: Int
    private var unitLower: [[Double]]
    private var diag: [Double]

    init(n: Int, bandwidth: Int, unitLower: [[Double]], diag: [Double]) {
        self.n = n
        self.bandwidth = bandwidth
        self.unitLower = unitLower
        self.diag = diag
    }

    /// Unit-factor entry (1 on the diagonal; zero outside the band).
    private func entry(row i: Int, col j: Int) -> Double {
        if i == j { return 1 }
        guard i > j, i - j <= bandwidth else { return 0 }
        return unitLower[i - j][i]
    }

    /// Solve `Ax = b` through the factor (forward, scale, back).
    func solve(_ b: [Double]) -> [Double] {
        precondition(b.count == n, "BandedCholesky.solve requires n right-hand values")
        var z = b
        for i in 0..<n {
            for j in max(0, i - bandwidth)..<i {
                z[i] -= unitLower[i - j][i] * z[j]
            }
        }
        for i in 0..<n {
            z[i] /= diag[i]
        }
        var x = z
        for i in stride(from: n - 1, through: 0, by: -1) {
            if i + 1 <= min(n - 1, i + bandwidth) {
                for j in (i + 1)...min(n - 1, i + bandwidth) {
                    x[i] -= unitLower[j - i][j] * x[j]
                }
            }
        }
        return x
    }

    /// Band of the inverse in `lower[d][i]` storage: Takahashi equations
    /// over the unit factor, O(n·b²) —
    /// `Z[i][j] = δ/D[i] − Σ_{k>i} L[k][i]·Z[k][j]` for `j ≥ i`
    /// (the diagonal scale applies to the Kronecker term only).
    /// Reads stay in-band by construction — both indices of every
    /// `Z[k][j]` read land in the current `[i, i+b]` window (rows above
    /// are complete, current-row entries to the right are computed
    /// first via descending `j`).
    func inverseBand() -> [[Double]] {
        var inv = [[Double]](repeating: [Double](repeating: 0, count: n), count: bandwidth + 1)
        func read(_ k: Int, _ j: Int) -> Double {
            let d = abs(k - j)
            guard d <= bandwidth else { return 0 }
            return inv[d][max(k, j)]
        }
        for i in stride(from: n - 1, through: 0, by: -1) {
            for j in stride(from: min(n - 1, i + bandwidth), through: i, by: -1) {
                var s = (i == j) ? 1.0 / diag[i] : 0.0
                if i + 1 <= min(n - 1, i + bandwidth) {
                    for k in (i + 1)...min(n - 1, i + bandwidth) {
                        s -= entry(row: k, col: i) * read(k, j)
                    }
                }
                inv[j - i][j] = s
            }
        }
        return inv
    }
}
