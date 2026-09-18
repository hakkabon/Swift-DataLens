import Foundation
import Testing
@testable import DataLens

private struct SolverFixture: Decodable {
    struct Case: Decodable {
        let name: String
        let matrix: [[Double]]
        let response: [Double]
        let expected: [Double]?
        let positiveDefinite: Bool
    }

    let schemaVersion: Int
    let tolerance: Double
    let cases: [Case]
}

@Suite("Shared solver conformance")
struct SolverConformanceTests {
    @Test func fallbackAndNumericCoreSeamMatchContract() throws {
        let url = try #require(Bundle.module.url(forResource: "solver-conformance",
                                                 withExtension: "json",
                                                 subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(SolverFixture.self, from: Data(contentsOf: url))
        #expect(fixture.schemaVersion == 1)

        for item in fixture.cases {
            let actual = Regression.solve(item.matrix, item.response)
            if let expected = item.expected {
                let solved = try #require(actual, "missing solution for \(item.name)")
                #expect(solved.count == expected.count)
                for (lhs, rhs) in zip(solved, expected) {
                    #expect(abs(lhs - rhs) <= fixture.tolerance, "mismatch for \(item.name)")
                }
                if item.positiveDefinite {
                    let spd = try #require(Regression.solveSPD(item.matrix, item.response))
                    for (lhs, rhs) in zip(spd, expected) {
                        #expect(abs(lhs - rhs) <= fixture.tolerance, "SPD mismatch for \(item.name)")
                    }
                }
            } else {
                #expect(actual == nil, "expected singular verdict for \(item.name)")
            }
        }
    }
}
