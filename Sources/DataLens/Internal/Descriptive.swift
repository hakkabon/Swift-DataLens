import Foundation

/// Vendored subset of Numerical-Statistics' `Descriptive` (byte-identical,
/// demoted to internal): `median` only, the robustness-scale statistic behind
/// the LOESS bisquare reweighting and its y-scale floor.
enum Descriptive {
    static func median(_ data: [Double]) -> Double? {
        guard !data.isEmpty else { return nil }
        let s = data.sorted()
        let n = s.count
        if n % 2 == 1 { return s[n / 2] }
        // Convert before adding to avoid integer overflow in generic callers.
        return (s[n / 2 - 1] + s[n / 2]) / 2
    }
}
