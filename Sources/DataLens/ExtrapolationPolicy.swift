import Foundation

/// What predictions and standard errors do outside the training bounding
/// box (per-dimension [min, max] over the kept training rows — a box, not
/// the convex hull).
///
/// This is a query-time concern only: the fit itself is identical under
/// every policy, so switching policies never refits. The default
/// (`polynomial`) preserves historical behavior exactly.
public enum ExtrapolationPolicy: Sendable {
    /// Extend the local polynomial fit (status-quo behavior; degree-2
    /// polynomials can grow quickly far outside the data).
    case polynomial
    /// Return the nearest training point's fitted value (and its standard
    /// error): flat, bounded, never surprising.
    case nearest
    /// No prediction outside the box (`.nan` from predict, nil from SE).
    case unavailable
}
