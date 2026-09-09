import Foundation

/// Seedable random number generators and `RandomNumberGenerator` helpers.
///
/// Vendored subset of Numerical-Statistics' `SeededRNG.swift` (byte-identical
/// streams, demoted to internal): test-only support for deterministic
/// smoothing tests. Promote to public API only with a `docs/DECISIONS.md`
/// entry — `DataLens` itself ships no samplers yet.
///
/// - `SplitMix64` — fast splittable generator, good for seeding and simple use.
/// - `Xoshiro256StarStar` — high-quality general-purpose generator.
/// - `SeedableRandomNumberGenerator` — backwards-compatible name wrapping
///   `Xoshiro256StarStar`.
///
/// All generators conform to `RandomNumberGenerator` and `Sendable`.

@inline(__always) private func rotl(_ x: UInt64, _ k: UInt64) -> UInt64 {
    (x << k) | (x >> (64 - k))
}

/// SplitMix64 — a fast, seedable 64-bit generator.
///
/// Useful directly for simple sampling and for seeding ``Xoshiro256StarStar``.
struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    init(seed: UInt64) {
        // Avoid the single fixed point of the all-zero state.
        self.state = seed &+ 0x9E3779B97F4A7C15
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

/// Xoshiro256** — a high-quality seedable generator (Blackman & Vigna, 2018).
///
/// Period 2^256 − 1. Suitable as the default reproducible generator.
struct Xoshiro256StarStar: RandomNumberGenerator, Sendable {
    private var s0: UInt64
    private var s1: UInt64
    private var s2: UInt64
    private var s3: UInt64

    /// Create a generator from a 64-bit seed (expanded via SplitMix64).
    init(seed: UInt64) {
        var seeder = SplitMix64(seed: seed)
        s0 = seeder.next()
        s1 = seeder.next()
        s2 = seeder.next()
        s3 = seeder.next()
        // All-zero state is invalid; nudge it deterministically.
        if s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 {
            s3 = 0x9E3779B97F4A7C15
        }
    }

    init<S: RandomNumberGenerator>(seededBy seeder: inout S) {
        s0 = seeder.next()
        s1 = seeder.next()
        s2 = seeder.next()
        s3 = seeder.next()
        if s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 {
            s3 = 0x9E3779B97F4A7C15
        }
    }

    mutating func next() -> UInt64 {
        let result = rotl(s1 &* 5, 7) &* 9
        let t = s1 << 17
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = rotl(s3, 45)
        return result
    }
}

/// Backwards-compatible seedable generator name.
///
/// The original sketch referenced `SeedableRandomNumberGenerator(seed: 42)`.
/// This is a thin wrapper over ``Xoshiro256StarStar``.
struct SeedableRandomNumberGenerator: RandomNumberGenerator, Sendable {
    private var inner: Xoshiro256StarStar

    init(seed: UInt64) {
        inner = Xoshiro256StarStar(seed: seed)
    }

    init<S: RandomNumberGenerator>(seededBy seeder: inout S) {
        inner = Xoshiro256StarStar(seededBy: &seeder)
    }

    mutating func next() -> UInt64 {
        inner.next()
    }
}

/// Cache for the spare deviate of the Marsaglia polar / Box–Muller transform.
///
/// Generating a standard normal pair yields two independent deviates; the
/// classic implementation discards one. Keep a ``GaussianCache`` alongside a
/// generator to reuse the spare (≈2× fewer uniform draws).
struct GaussianCache: Sendable {
    private var spare: Double?

    init() {}

    /// Next standard-normal deviate, reusing the cached spare when available.
    mutating func nextStandardNormal<R: RandomNumberGenerator>(using rng: inout R) -> Double {
        if let s = spare {
            spare = nil
            return s
        }
        var x1 = 0.0, x2 = 0.0, w = 0.0
        repeat {
            x1 = 2.0 * Double.random(in: 0..<1, using: &rng) - 1.0
            x2 = 2.0 * Double.random(in: 0..<1, using: &rng) - 1.0
            w = x1 * x1 + x2 * x2
        } while w >= 1.0 || w == 0.0
        let factor = sqrt(-2.0 * log(w) / w)
        spare = x2 * factor
        return x1 * factor
    }
}

extension RandomNumberGenerator {
    /// Uniform `Double` in a half-open range.
    mutating func nextDouble(in range: Range<Double>) -> Double {
        Double.random(in: range, using: &self)
    }

    /// Uniform `Double` in a closed range.
    mutating func nextDouble(in range: ClosedRange<Double>) -> Double {
        Double.random(in: range, using: &self)
    }

    /// Uniform `Int` in a half-open range.
    mutating func nextInt(in range: Range<Int>) -> Int {
        Int.random(in: range, using: &self)
    }

    /// Bernoulli trial with success probability `p`.
    mutating func nextBool(probability p: Double) -> Bool {
        precondition(p >= 0 && p <= 1, "probability must be in [0, 1]")
        return Double.random(in: 0..<1, using: &self) < p
    }
}
