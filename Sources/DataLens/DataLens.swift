/// Local-regression toolkit: Cleveland-style LOESS, then clean-room
/// Loader-style adaptive smoothing and local likelihood.
///
/// `Loess` is a byte-identical port of Numerical-Statistics' `Loess.swift`
/// over minimal vendored numerics (`Internal/`); no GPL `locfit` code enters
/// here.
public enum DataLens {
    /// Library version marker.
    public static let version = "0.1.0"
}
