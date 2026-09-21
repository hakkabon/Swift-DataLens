// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Swift-DataLens",
    platforms: [
        .macOS(.v13), .iOS(.v16), .tvOS(.v16), .watchOS(.v9), .macCatalyst(.v16),
    ],
    products: [
        .library(name: "DataLens", targets: ["DataLens"]),
    ],
    dependencies: [
        // 0.7.0 consumes the Rust-NumericCore 0.5.0 checksummed framework,
        // including the sparse statistical-solve ABI used by likelihood GAMs.
        // Keep a minor-series bound: new NumericCore behavior is opt-in here,
        // while compatible framework and binding refreshes remain resolvable.
        .package(url: "https://github.com/hakkabon/Swift-NumericCore.git",
                 .upToNextMinor(from: "0.7.0")),
    ],
    targets: [
        .target(
            name: "DataLens",
            dependencies: [
                // Apple-only (imports Accelerate, and its XCFramework has
                // no Linux slice); Linux builds use the vendored fallback
                // via `#if canImport(NumericCoreAccelerate)`.
                .product(
                    name: "NumericCore",
                    package: "Swift-NumericCore",
                    condition: .when(platforms: [.macOS, .iOS, .macCatalyst, .tvOS, .watchOS, .visionOS])
                ),
                .product(
                    name: "NumericCoreAccelerate",
                    package: "Swift-NumericCore",
                    condition: .when(platforms: [.macOS, .iOS, .macCatalyst, .tvOS, .watchOS, .visionOS])
                ),
                // Portable CSR CGLS is selected only for profiled sparse
                // multivariate workloads; dense QR remains the small/dense path.
                .product(
                    name: "NumericCoreSparse",
                    package: "Swift-NumericCore",
                    condition: .when(platforms: [.macOS, .iOS, .macCatalyst, .tvOS, .watchOS, .visionOS])
                ),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
        .testTarget(
            name: "DataLensTests",
            dependencies: ["DataLens"],
            resources: [.copy("Fixtures")]
        ),
        .executableTarget(
            name: "Benchmarks",
            dependencies: ["DataLens"],
            path: "Benchmarks"
        ),
    ]
)
