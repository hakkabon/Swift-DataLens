import Foundation
import Testing
@testable import DataLens

/// Banded SPD solver pinned against dense textbook references: the
/// Whittaker smoother trusts this factor for fits, traces, and SEs, so
/// every entry is checked, not just residuals.
@Suite("Banded SPD")
struct BandedSPDTests {
    /// Textbook dense Cholesky (lower), nil on non-positive pivots.
    func denseCholesky(_ a: [[Double]]) -> [[Double]]? {
        let n = a.count
        var l = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for j in 0...i {
                var s = a[i][j]
                for k in 0..<j { s -= l[i][k] * l[j][k] }
                if i == j {
                    guard s > 0 else { return nil }
                    l[i][j] = sqrt(s)
                } else {
                    l[i][j] = s / l[j][j]
                }
            }
        }
        return l
    }

    func denseSolve(l: [[Double]], b: [Double]) -> [Double] {
        let n = b.count
        var z = b
        for i in 0..<n {
            for j in 0..<i { z[i] -= l[i][j] * z[j] }
            z[i] /= l[i][i]
        }
        var x = z
        for i in stride(from: n - 1, through: 0, by: -1) {
            for j in (i + 1)..<n { x[i] -= l[j][i] * x[j] }
            x[i] /= l[i][i]
        }
        return x
    }

    /// Dense inverse via Gauss-Jordan (test-only, tiny n).
    func denseInverse(_ a: [[Double]]) -> [[Double]]? {
        let n = a.count
        var m = a
        var inv = (0..<n).map { i in (0..<n).map { $0 == i ? 1.0 : 0.0 } }
        for c in 0..<n {
            var p = c
            for r in c..<n where abs(m[r][c]) > abs(m[p][c]) { p = r }
            guard m[p][c] != 0 else { return nil }
            m.swapAt(c, p); inv.swapAt(c, p)
            let d = m[c][c]
            for j in 0..<n { m[c][j] /= d; inv[c][j] /= d }
            for r in 0..<n where r != c {
                let f = m[r][c]
                for j in 0..<n { m[r][j] -= f * m[c][j]; inv[r][j] -= f * inv[c][j] }
            }
        }
        return inv
    }

    /// Random SPD: M·Mᵀ + n·I with a fixed seed.
    func randomSPD(n: Int, seed: UInt64) -> [[Double]] {
        var rng = SeedableRandomNumberGenerator(seed: seed)
        let m = (0..<n).map { _ in (0..<n).map { _ in Double.random(in: -1...1, using: &rng) } }
        return (0..<n).map { i in
            (0..<n).map { j in
                (0..<n).reduce(0.0) { $0 + m[i][$1] * m[j][$1] } + (i == j ? Double(n) : 0)
            }
        }
    }

    func maxDiff(_ a: [[Double]], _ b: [[Double]]) -> Double {
        zip(a, b).reduce(0.0) { acc, pair in
            acc + zip(pair.0, pair.1).reduce(0.0) { $0 + abs($1.0 - $1.1) }
        }
    }

    @Test func factorSolveAndInverseMatchDense() {
        for (n, seed) in [(2, UInt64(9001)), (5, UInt64(9002)), (10, UInt64(9003))] {
            let a = randomSPD(n: n, seed: seed)
            let band = BandedMatrix(n: n, bandwidth: n - 1) { i, j in a[i][j] }
            guard let factor = band.cholesky() else {
                Issue.record("cholesky failed on SPD n=\(n)")
                continue
            }
            let ref = denseCholesky(a)!
            // Assembly reads back exactly.
            var diff = 0.0
            for i in 0..<n {
                for j in 0...i {
                    diff = max(diff, abs(band.value(row: i, col: j) - a[i][j]))
                }
            }
            #expect(diff == 0)  // assembly reads back exactly
            // Solves against the dense reference.
            let b = (0..<n).map { Double($0) + 0.5 }
            let x = factor.solve(b)
            let xr = denseSolve(l: ref, b: b)
            #expect(zip(x, xr).map { abs($0 - $1) }.max()! <= 1e-9)
            // Inverse band against the dense inverse (full band here).
            let inv = factor.inverseBand()
            let invRef = denseInverse(a)!
            var bandDiff = 0.0
            for i in 0..<n {
                for d in 0...(n - 1 - i) {
                    bandDiff = max(bandDiff, abs(inv[d][i + d] - invRef[i + d][i]))
                }
            }
            #expect(bandDiff <= 1e-9)
        }
    }

    @Test func narrowBandMatchesBandedInput() {
        // Genuinely banded input (tridiagonal + pentadiagonal): the band
        // solver must reproduce the dense reference with no truncation.
        var rng = SeedableRandomNumberGenerator(seed: 9004)
        let n = 12
        var a = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n {
            a[i][i] = 4 + Double.random(in: 0...1, using: &rng)
            if i + 1 < n {
                let v = Double.random(in: -1...1, using: &rng)
                a[i][i + 1] = v; a[i + 1][i] = v
            }
            if i + 2 < n {
                let v = 0.5 * Double.random(in: -1...1, using: &rng)
                a[i][i + 2] = v; a[i + 2][i] = v
            }
        }
        let band = BandedMatrix(n: n, bandwidth: 2) { i, j in a[i][j] }
        guard let factor = band.cholesky() else {
            Issue.record("cholesky failed on banded SPD")
            return
        }
        let ref = denseCholesky(a)!
        let b = (0..<n).map { sin(Double($0)) }
        let x = factor.solve(b)
        let xr = denseSolve(l: ref, b: b)
        #expect(zip(x, xr).map { abs($0 - $1) }.max()! <= 1e-9)
        let inv = factor.inverseBand()
        let invRef = denseInverse(a)!
        var bandDiff = 0.0
        for i in 0..<n {
            for d in 0...min(2, n - 1 - i) {
                bandDiff = max(bandDiff, abs(inv[d][i + d] - invRef[i + d][i]))
            }
        }
        #expect(bandDiff <= 1e-9)
    }

    @Test func rejectsNonSPD() {
        // Zero pivot and indefinite inputs return nil (verdict parity
        // with solveSPD), never a factor.
        let zero = BandedMatrix(n: 2, bandwidth: 1) { i, j in i == j ? (i == 0 ? 0 : 1) : 0 }
        #expect(zero.cholesky() == nil)
        let indef = BandedMatrix(n: 2, bandwidth: 1) { i, j in i == j ? 1 : 2 }
        #expect(indef.cholesky() == nil)
        // n = 1 trivially factors.
        let one = BandedMatrix(n: 1, bandwidth: 0) { _, _ in 4 }
        #expect(one.cholesky() != nil)
        #expect(one.cholesky()!.solve([2]) == [0.5])
    }
}
