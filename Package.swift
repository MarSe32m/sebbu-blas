// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "sebbu-blas",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "SebbuBLAS", targets: ["SebbuBLAS"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/MarSe32m/sebbu-copenblas",
            .upToNextMinor(from: "0.3.34")
        ),
        .package(
            url: "https://github.com/apple/swift-numerics",
            .upToNextMajor(from: "1.1.1")
        ),
    ],
    targets: [
        .target(
            name: "SebbuBLAS",
            dependencies: [
                .product(
                    name: "COpenBLAS",
                    package: "sebbu-copenblas",
                    condition: .when(platforms: [.linux, .windows])
                ),
                .product(name: "ComplexModule", package: "swift-numerics"),
                .product(name: "RealModule", package: "swift-numerics")
            ],
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK", .when(platforms: [.macOS])),
                .define("ACCELERATE_LAPACK_ILP64", .when(platforms: [.macOS])),
            ],
            swiftSettings: [
                .define("SEBBU_BLAS_FORCE_SWIFT")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.macOS])),
            ]
        ),
        .testTarget(
            name: "SebbuBLASTests",
            dependencies: [
                "SebbuBLAS",
                .product(name: "ComplexModule", package: "swift-numerics"),
                .product(name: "RealModule", package: "swift-numerics")
            ]
        ),
    ]
)
