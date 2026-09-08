// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Swift-Data-Lens",
    platforms: [
        .macOS(.v13), .iOS(.v16), .tvOS(.v16), .watchOS(.v9), .macCatalyst(.v16),
    ],
    products: [
        .library(name: "Data-Lens", targets: ["Swift-Data-Lens"]),
    ],
    targets: [
        .target(
            name: "Swift-Data-Lens"),
        .testTarget(
            name: "Swift-LensTests",
            dependencies: ["Swift-Data-Lens"]
        ),
    ]
)
