import Foundation

/// Indexed parallel map over `0..<count` via a throwing task group.
///
/// Per-point smoother work is independent and `Sendable`-clean, so batch
/// evaluation parallelizes with zero arithmetic change: each index runs the
/// same code as the sequential loop, and results are re-sorted into index
/// order — concurrent batches are bit-identical to sequential ones (pinned
/// by `BatchTests`). Reductions across points (residual medians, traces,
/// deviances) stay sequential in the callers, in index order.
///
/// Cooperative cancellation: each task checks for cancellation before
/// starting its point, so cancelling the caller's task aborts pending
/// points with `CancellationError`. Points already running finish first —
/// CPU work cannot be preempted — so cancellation bounds wasted work
/// rather than stopping it instantly.
func concurrentMap<T: Sendable>(over count: Int,
                                _ work: @Sendable @escaping (Int) -> T) async throws -> [T] {
    guard count > 0 else { return [] }
    return try await withThrowingTaskGroup(of: (Int, T).self) { group in
        for i in 0..<count {
            group.addTask {
                try Task.checkCancellation()
                return (i, work(i))
            }
        }
        var out: [(Int, T)] = []
        out.reserveCapacity(count)
        for try await r in group { out.append(r) }
        return out.sorted { $0.0 < $1.0 }.map { $0.1 }
    }
}
