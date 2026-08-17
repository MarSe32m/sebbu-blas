// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

#if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
import COpenBLAS
#elseif (canImport(Accelerate) && os(macOS)) && !SEBBU_BLAS_FORCE_SWIFT
import Accelerate
#endif

/// A namespace for level 1, 2, and 3 BLAS operations.
@frozen
public enum BLAS {}

extension BLAS {
    public typealias Index = Int

    @frozen
    public enum Layout: Int, Sendable {
        case rowMajor = 101
        case columnMajor = 102
    }

    @frozen
    public enum Transpose: Int, Sendable {
        case noTranspose = 111
        case transpose = 112
        case conjugateTranspose = 113
        case conjugateNoTranspose = 114
    }

    @frozen
    public enum Triangle: Int, Sendable {
        case upper = 121
        case lower = 122
    }

    @frozen
    public enum Diagonal: Int, Sendable {
        case nonUnit = 131
        case unit = 132
    }

    @frozen
    public enum Side: Int, Sendable {
        case left = 141
        case right = 142
    }
}

#if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
@inlinable
@inline(always)
@_transparent
internal func _backendIndex(_ value: Int) -> Int32 {
    precondition(
        value >= Int(Int32.min) && value <= Int(Int32.max),
        "The OpenBLAS LP64 interface requires every integer argument to fit in Int32"
    )
    return Int32(value)
}
#elseif (canImport(Accelerate) && os(macOS)) && !SEBBU_BLAS_FORCE_SWIFT
@inlinable
@inline(always)
@_transparent
internal func _backendIndex(_ value: Int) -> Int {
    value
}
#endif

#if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
extension BLAS.Layout {
    @usableFromInline
    internal var _cblas: CBLAS_ORDER {
        switch self {
        case .rowMajor: CblasRowMajor
        case .columnMajor: CblasColMajor
        }
    }
}

extension BLAS.Transpose {
    @usableFromInline
    internal var _cblas: CBLAS_TRANSPOSE {
        switch self {
        case .noTranspose: CblasNoTrans
        case .transpose: CblasTrans
        case .conjugateTranspose: CblasConjTrans
        case .conjugateNoTranspose:
            #if canImport(COpenBLAS)
            CblasConjNoTrans
            #else
            AtlasConj
            #endif
        }
    }
}

extension BLAS.Triangle {
    @usableFromInline
    internal var _cblas: CBLAS_UPLO {
        switch self {
        case .upper: CblasUpper
        case .lower: CblasLower
        }
    }
}

extension BLAS.Diagonal {
    @usableFromInline
    internal var _cblas: CBLAS_DIAG {
        switch self {
        case .nonUnit: CblasNonUnit
        case .unit: CblasUnit
        }
    }
}

extension BLAS.Side {
    @usableFromInline
    internal var _cblas: CBLAS_SIDE {
        switch self {
        case .left: CblasLeft
        case .right: CblasRight
        }
    }
}
#endif
