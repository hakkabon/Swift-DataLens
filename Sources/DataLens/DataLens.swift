/// Local-regression toolkit: Cleveland-style LOESS, plus clean-room
/// Loader-style adaptive smoothing; local likelihood next.
///
/// `Loess` behavior is frozen (see `docs/DECISIONS.md`); new smoothers land
/// as new types. No GPL `locfit` code enters here.
public enum DataLens {
    /// Library version marker.
    public static let version = "0.2.0"
}
