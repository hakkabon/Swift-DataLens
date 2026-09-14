/// Local-regression toolkit: Cleveland-style LOESS, clean-room adaptive
/// smoothing, local likelihood, and batch/concurrent evaluation.
///
/// Behaviors are frozen (see `docs/DECISIONS.md`); new smoothers land as new
/// types. No GPL `locfit` code enters here.
public enum DataLens {
    /// Library version marker.
    public static let version = "0.4.0"
}
