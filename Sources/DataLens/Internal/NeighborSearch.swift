import Foundation

/// Neighbor lookup over a fixed training matrix: kd-tree when it pays,
/// brute force otherwise. Output is exactly ``Loess/nearestIndices(_:to:count:)``
/// on every path (proven by `NearestNeighborTests`).
///
/// Build once per fit and thread through the local fits: the tree costs
/// O(n log n) to build, so rebuilding it per query would be slower than
/// brute force. Single-query callers (`predict`, `standardError`) build one
/// per call; a batch-predict API can reuse this type directly later.
struct NeighborSearch: Sendable {
    /// Below this row count the tree build never amortizes, even across a fit.
    private static let treeThreshold = 64
    /// The tree prunes only when the neighborhood is a fraction of n: filling
    /// the queue already costs O(n·k), so near-n neighborhoods visit the whole
    /// tree and lose to one brute-force sort. Heuristic; recalibrate with
    /// release-mode benches if spans drift.
    private static let treeFraction = 4

    private let trainX: [[Double]]
    private let tree: KDTree?

    /// Training rows (for basis evaluation and bandwidths).
    var trainingPoints: [[Double]] { trainX }
    /// Whether a tree was built (test hook; path choice is per query).
    var treeBuilt: Bool { tree != nil }

    /// - Parameter forBatchUse: false for single-shot queries (`predict`,
    ///   `standardError`), where the O(n log² n) build exceeds one brute scan.
    init(trainX: [[Double]], forBatchUse: Bool = true) {
        self.trainX = trainX
        let dims = trainX.first?.count ?? 0
        if forBatchUse, trainX.count >= Self.treeThreshold, dims > 0 {
            self.tree = KDTree(points: trainX.indices.map { KDPoint(index: $0, coords: trainX[$0]) })
        } else {
            self.tree = nil
        }
    }

    /// Indices of the `count` nearest training rows to `x`.
    /// Returns `min(max(count,1), n)` indices, sorted nearest-first.
    func nearest(to x: [Double], count: Int) -> [Int] {
        let k = min(max(count, 1), trainX.count)
        guard k > 0 else { return [] }
        if let tree, k * Self.treeFraction <= trainX.count {
            return tree.nearestIndices(to: x, count: k)
        }
        return Loess.nearestIndices(trainX, to: x, count: k)
    }
}
