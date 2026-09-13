import Foundation

/// N-dimensional kd-tree for exact k-nearest-neighbor lookup.
///
/// Generalized to N-D (plus exact-parity fixes) from a `Point2D`
/// implementation provided by U. Akerstedt-Inoue; vendored here so
/// `DataLens` builds dependency-free. Output matches
/// ``Loess/nearestIndices(_:to:count:)`` exactly: ascending distance,
/// lower-index-first on ties.
struct KDPoint: Sendable {
    /// Original row index in the training matrix.
    let index: Int
    let coords: [Double]

    @inlinable
    func squaredDistance(to other: [Double]) -> Double {
        precondition(coords.count == other.count, "kd-tree query must match tree dimensionality")
        var d = 0.0
        for (a, b) in zip(coords, other) {
            let delta = a - b
            d += delta * delta
        }
        return d
    }
}

/// Bounded result queue ordered lexicographically by (distance, index).
///
/// The index tie-break is what makes tree output exactly equal brute force
/// (which stable-sorts distance over index-ordered rows). Without it,
/// equidistant points come out in traversal order.
struct BoundedNeighborQueue: Sendable {
    let maxCapacity: Int
    private(set) var elements: [(index: Int, squaredDistance: Double)] = []

    var isFull: Bool { elements.count >= maxCapacity }
    var maxDistanceSq: Double { elements.last?.squaredDistance ?? Double.infinity }

    init(maxCapacity: Int) {
        self.maxCapacity = maxCapacity
    }

    @inlinable
    static func before(_ a: (index: Int, squaredDistance: Double),
                       _ b: (index: Int, squaredDistance: Double)) -> Bool {
        if a.squaredDistance != b.squaredDistance { return a.squaredDistance < b.squaredDistance }
        return a.index < b.index
    }

    mutating func insert(index: Int, squaredDistance: Double) {
        let candidate = (index, squaredDistance)
        // O(k) ordered insert: the queue is already sorted, so a full sort
        // per insert would cost O(k log k) each — dominant at large k.
        if !isFull {
            if let i = elements.firstIndex(where: { Self.before(candidate, $0) }) {
                elements.insert(candidate, at: i)
            } else {
                elements.append(candidate)
            }
        } else if let last = elements.last, Self.before(candidate, last) {
            elements.removeLast()
            if let i = elements.firstIndex(where: { Self.before(candidate, $0) }) {
                elements.insert(candidate, at: i)
            } else {
                elements.append(candidate)
            }
        }
    }
}

struct KDTree: Sendable {
    private enum Node: Sendable {
        case empty
        indirect case node(point: KDPoint, splitAxis: Int, left: Node, right: Node)
    }

    private let root: Node

    init(points: [KDPoint]) {
        precondition(points.allSatisfy({ $0.coords.count == points.first?.coords.count ?? 0 }),
                     "kd-tree points must share one dimensionality")
        let dims = points.first?.coords.count ?? 0
        self.root = dims > 0 ? Self.build(points, axis: 0, dims: dims) : .empty
    }

    private static func build(_ points: [KDPoint], axis: Int, dims: Int) -> Node {
        guard !points.isEmpty else { return .empty }
        // Stable sort keeps index order among equal coordinates.
        let sorted = points.sorted { $0.coords[axis] < $1.coords[axis] }
        let m = sorted.count / 2
        let next = (axis + 1) % dims
        return .node(point: sorted[m], splitAxis: axis,
                     left: build(Array(sorted[..<m]), axis: next, dims: dims),
                     right: build(Array(sorted[(m + 1)...]), axis: next, dims: dims))
    }

    /// Indices of the `count` nearest points, sorted nearest-first.
    /// Caller clamps `count` to `min(max(k,1), n)`; empty tree yields `[]`.
    func nearestIndices(to query: [Double], count k: Int) -> [Int] {
        guard k > 0 else { return [] }
        var queue = BoundedNeighborQueue(maxCapacity: k)
        search(root, query: query, queue: &queue)
        return queue.elements.map(\.index)
    }

    private func search(_ node: Node, query: [Double], queue: inout BoundedNeighborQueue) {
        switch node {
        case .empty:
            return
        case let .node(point, axis, left, right):
            queue.insert(index: point.index, squaredDistance: point.squaredDistance(to: query))
            let delta = query[axis] - point.coords[axis]
            let (first, second) = delta < 0 ? (left, right) : (right, left)
            search(first, query: query, queue: &queue)
            // `<=` (not `<`): the far side can hold equidistant points with
            // smaller indices, which brute force would include.
            if !queue.isFull || delta * delta <= queue.maxDistanceSq {
                search(second, query: query, queue: &queue)
            }
        }
    }
}
