import Foundation
import Testing
@testable import DataLens

/// Exact agreement between the kd-tree path (`NeighborSearch`) and the
/// brute-force reference (`Loess.nearestIndices`): same indices, same order,
/// on every path (tree and small-n fallback).
@Suite("Nearest neighbors")
struct NearestNeighborTests {
    @Test func treeMatchesBruteForce() {
        for dims in [1, 2, 3, 5] {
            for n in [1, 2, 5, 40, 120, 300] {
                var rng = SeedableRandomNumberGenerator(seed: 1000 + UInt64(dims * 256 + n))
                // Quantized coordinates guarantee duplicate rows and ties.
                let trainX = (0..<n).map { _ in
                    (0..<dims).map { _ in (Double.random(in: 0..<4, using: &rng) * 2).rounded() / 2 }
                }
                let search = NeighborSearch(trainX: trainX)
                // The tree path is only built (and only taken) at large n
                // with small k; assert the hook matches the routing rule.
                #expect(search.treeBuilt == (n >= 64))
                for _ in 0..<5 {
                    let q = (0..<dims).map { _ in Double.random(in: -1...4, using: &rng) }
                    for k in [1, 2, 5, n, n + 3] {
                        #expect(search.nearest(to: q, count: k)
                            == Loess.nearestIndices(trainX, to: q, count: k))
                    }
                }
            }
        }
    }

    @Test func duplicatesAndCoincidentQuery() {
        // Duplicate rows coinciding with the query: the strict-`<` backtrack
        // would miss the smaller-index copy; `<=` must not.
        let trainX = [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        let search = NeighborSearch(trainX: trainX)
        for k in 1...6 {
            #expect(search.nearest(to: [1.0, 1.0], count: k)
                == Loess.nearestIndices(trainX, to: [1.0, 1.0], count: k))
        }
        #expect(search.nearest(to: [1.0, 1.0], count: 2) == [0, 1])
        #expect(search.nearest(to: [1.0, 1.0], count: 3) == [0, 1, 3])
    }

    @Test func emptyAndDegenerate() {
        #expect(NeighborSearch(trainX: []).nearest(to: [0.0], count: 3) == [])
        #expect(NeighborSearch(trainX: [[Double]()]).nearest(to: [], count: 1) == [0])
        #expect(NeighborSearch(trainX: [[], [], []]).nearest(to: [], count: 5) == [0, 1, 2])
    }

    @Test func mixedPathsAgree() {
        // n above the build threshold with k on both sides of the routing
        // rule (tree iff k*4 <= n): agreement must hold on every path.
        var rng = SeedableRandomNumberGenerator(seed: 77)
        for n in [200, 300] {
            let trainX = (0..<n).map { _ in [Double.random(in: 0...5, using: &rng),
                                              Double.random(in: 0...5, using: &rng)] }
            let search = NeighborSearch(trainX: trainX)
            #expect(search.treeBuilt)
            for _ in 0..<10 {
                let q = [Double.random(in: 0...5, using: &rng),
                         Double.random(in: 0...5, using: &rng)]
                for k in [1, 10, n / 4, n / 2, n] {
                    #expect(search.nearest(to: q, count: k)
                        == Loess.nearestIndices(trainX, to: q, count: k))
                }
            }
        }
        // Single-shot searches never build.
        #expect(!NeighborSearch(trainX: [[0.0], [1.0]], forBatchUse: false).treeBuilt)
    }
}
