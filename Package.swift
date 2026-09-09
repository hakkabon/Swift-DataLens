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
    targets: [
        .target(
            name: "DataLens",
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
