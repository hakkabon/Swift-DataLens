import Foundation

/// Missing-data handling shared by the smoothers' `droppingMissing` paths:
/// rows with a non-finite coordinate or response are dropped, preserving
/// order, and survivors' original indices are reported (so fitted values
/// stay joinable to the caller's table).
enum MissingData {
    static func dropping(trainX: [[Double]], trainY: [Double])
        -> (trainX: [[Double]], trainY: [Double], kept: [Int])
    {
        precondition(trainX.count == trainY.count, "trainX and trainY must have equal counts")
        var fx: [[Double]] = []
        var fy: [Double] = []
        var kept: [Int] = []
        fx.reserveCapacity(trainX.count)
        fy.reserveCapacity(trainY.count)
        kept.reserveCapacity(trainX.count)
        for (i, (x, y)) in zip(trainX, trainY).enumerated() {
            if y.isFinite, x.allSatisfy({ $0.isFinite }) {
                fx.append(x)
                fy.append(y)
                kept.append(i)
            }
        }
        return (fx, fy, kept)
    }
}
