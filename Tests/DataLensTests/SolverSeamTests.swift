import Foundation
import Testing
@testable import DataLens

/// Direct coverage of the solver seam's nil contract and correctness, on
/// whichever path is active (LAPACK via Accelerate on Apple platforms,
/// vendored Householder/elimination fallback elsewhere).
@Suite("Solver seam")
struct SolverSeamTests {
    @Test func leastSquaresSolves() {
        // Through-origin line: β = Σxy/Σx² = 28/14 = 2.
        let beta = LinAlg.leastSquares(design: [[1], [2], [3]], response: [2, 4, 6])!
        #expect(beta.count == 1)
        #expect(abs(beta[0] - 2.0) <= 1e-9)
    }

    @Test func leastSquaresRankDeficientIsNil() {
        // Duplicate columns: rank 1 in a 3-wide... two identical columns.
        #expect(LinAlg.leastSquares(design: [[1, 1], [2, 2], [3, 3]], response: [1, 2, 3]) == nil)
        // Wide design (m < n).
        #expect(LinAlg.leastSquares(design: [[1, 2, 3]], response: [1]) == nil)
        // Ragged / empty / mismatched.
        #expect(LinAlg.leastSquares(design: [[1], [2, 3]], response: [1, 2]) == nil)
        #expect(LinAlg.leastSquares(design: [], response: []) == nil)
        #expect(LinAlg.leastSquares(design: [[1]], response: [1, 2]) == nil)
    }

    @Test func squareSolveSolves() {
        // 2x + y = 5, x + 3y = 6 → (1.8, 1.4).
        let x = Regression.solve([[2, 1], [1, 3]], [5, 6])!
        #expect(x.count == 2)
        #expect(abs(x[0] - 1.8) <= 1e-9)
        #expect(abs(x[1] - 1.4) <= 1e-9)
    }

    @Test func squareSolveSingularIsNil() {
        #expect(Regression.solve([[1, 2], [2, 4]], [3, 6]) == nil)
        #expect(Regression.solve([[1, 2, 3], [4, 5, 6]], [1, 2]) == nil)
        #expect(Regression.solve([[1]], []) == nil)
    }

    @Test func spdSolveSolves() {
        // 4x + 2y = 8, 2x + 3y = 7 → (1.25, 1.5). Eigenvalues
        // (7±√17)/2 > 0, so SPD.
        let x = Regression.solveSPD([[4, 2], [2, 3]], [8, 7])!
        #expect(x.count == 2)
        #expect(abs(x[0] - 1.25) <= 1e-9)
        #expect(abs(x[1] - 1.5) <= 1e-9)
    }

    @Test func spdSolveNotDefiniteIsNil() {
        // Symmetric indefinite (eigenvalues 3, −1).
        #expect(Regression.solveSPD([[1, 2], [2, 1]], [3, 3]) == nil)
        // Zero matrix.
        #expect(Regression.solveSPD([[0, 0], [0, 0]], [0, 0]) == nil)
        // Non-square / mismatched.
        #expect(Regression.solveSPD([[4, 2, 1], [2, 3, 0]], [8, 7]) == nil)
        #expect(Regression.solveSPD([[4, 2], [2, 3]], [8]) == nil)
    }
}
