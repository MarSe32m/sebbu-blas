# sebbu-blas

`sebbu-blas` is a Swift namespace wrapper for the CBLAS API. It exposes the
`cblas_*` routines supplied in the OpenBLAS `cblas.h`, including the
OpenBLAS matrix-copy, matrix-addition and batched GEMM extensions.

```swift
import SebbuBLAS

let x = [1.0, 2.0, 3.0]
let y = [4.0, 5.0, 6.0]

let result = BLAS.ddot(n: x.count, x: x, incX: 1, y: y, incY: 1)
```

All integer parameters and index results are Swift `Int`. Before calling the
LP64 OpenBLAS interface, every integer passed to C is checked to fit in
`Int32`; an out-of-range value fails a precondition instead of truncating.

Complex functions use `Complex<Float>` and `Complex<Double>` from
swift-numerics, including all pointer and pointer-array parameters.

## Backends

- Linux and Windows use `COpenBLAS` from `sebbu-copenblas`.
- macOS uses Accelerate with its ILP64 interface. OpenBLAS-only extensions use
  their Swift implementations.
- Other platforms use the pure Swift implementations (with swift-numerics for
  complex values).
- Passing `-Xswiftc -DSEBBU_BLAS_FORCE_SWIFT` forces the Swift backend, which
  is useful for testing it on a platform that also has a native backend.

The Swift fallback implements dense, packed and banded level 1–3 operations.
Its GEMM path is cache tiled but deliberately contains no manual SIMD.

## Package dependency

```swift
.package(url: "https://github.com/your-name/sebbu-blas", from: "0.1.0")
```

Add `.product(name: "SebbuBLAS", package: "sebbu-blas")` to your target.

On platforms where `Accelerate` is the default backend, you need to add 
the following settings to your target
```swift
    cSettings: [
        .define("ACCELERATE_NEW_LAPACK", .when(platforms: [.macOS, .iOS, .watchOS, .tvOS])),
        .define("ACCELERATE_LAPACK_ILP64", .when(platforms: [.macOS, .iOS, .watchOS, .tvOS])),
    ],
    linkerSettings: [
        .linkedFramework("Accelerate", .when(platforms: [.macOS, .iOS, .watchOS, .tvOS])),
    ]
```

## Validation

Run the tests with both the selected platform backend and the Swift fallback:

```sh
swift test
swift test -Xswiftc -DSEBBU_BLAS_FORCE_SWIFT
```

The API is intentionally pointer based, like CBLAS. Callers are responsible
for providing buffers large enough for the dimensions, leading dimensions,
strides and batch counts they pass.
