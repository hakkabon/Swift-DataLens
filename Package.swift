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
        // Pinned to the tagged 0.1.0 commit: `from: "0.1.0"` fails to
        // resolve (see DECISIONS #9). Bump manually on new NumericCore tags.
        .package(url: "https://github.com/hakkabon/Swift-NumericCore.git",
                 revision: "c9892c7fc4a76dfefd3480dead9d3ddfaeb50567"),
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
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
        .testTarget(
            name: "DataLensTests",
            dependencies: ["DataLens"]
        ),
        .executableTarget(
            name: "Benchmarks",
            dependencies: ["DataLens"],
            path: "Benchmarks"
        ),
    ]
)
