// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

#if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
import COpenBLAS
#elseif (canImport(Accelerate) && os(macOS)) && !SEBBU_BLAS_FORCE_SWIFT
import Accelerate
#endif

import ComplexModule
import RealModule

extension BLAS {
    @inlinable
    @inline(always)
    public static func sgemm(layout: Layout, transposeA: Transpose, transposeB: Transpose, m: Int, n: Int, k: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, b: UnsafePointer<Float>, ldb: Int, beta: Float, c: UnsafeMutablePointer<Float>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sgemm(layout._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(k), alpha, a, _backendIndex(lda), b, _backendIndex(ldb), beta, c, _backendIndex(ldc))
        #else
        _gemm(layout: layout, transposeA: transposeA, transposeB: transposeB, m: m, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dgemm(layout: Layout, transposeA: Transpose, transposeB: Transpose, m: Int, n: Int, k: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, b: UnsafePointer<Double>, ldb: Int, beta: Double, c: UnsafeMutablePointer<Double>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dgemm(layout._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(k), alpha, a, _backendIndex(lda), b, _backendIndex(ldb), beta, c, _backendIndex(ldc))
        #else
        _gemm(layout: layout, transposeA: transposeA, transposeB: transposeB, m: m, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cgemm(layout: Layout, transposeA: Transpose, transposeB: Transpose, m: Int, n: Int, k: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafePointer<Complex<Float>>, ldb: Int, beta: Complex<Float>, c: UnsafeMutablePointer<Complex<Float>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_cgemm(layout._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(k), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(b), _backendIndex(ldb), _complexFloatPointer(beta), _complexFloatPointer(c), _backendIndex(ldc))
            }
        }
        #else
        _gemm(layout: layout, transposeA: transposeA, transposeB: transposeB, m: m, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zgemm(layout: Layout, transposeA: Transpose, transposeB: Transpose, m: Int, n: Int, k: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafePointer<Complex<Double>>, ldb: Int, beta: Complex<Double>, c: UnsafeMutablePointer<Complex<Double>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zgemm(layout._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(k), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(b), _backendIndex(ldb), _complexDoublePointer(beta), _complexDoublePointer(c), _backendIndex(ldc))
            }
        }
        #else
        _gemm(layout: layout, transposeA: transposeA, transposeB: transposeB, m: m, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc)
        #endif
    }
}

// MARK: - Symmetric and Hermitian matrix multiplication

extension BLAS {
    @inlinable
    @inline(always)
    public static func ssymm(layout: Layout, side: Side, triangle: Triangle, m: Int, n: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, b: UnsafePointer<Float>, ldb: Int, beta: Float, c: UnsafeMutablePointer<Float>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ssymm(layout._cblas, side._cblas, triangle._cblas, _backendIndex(m), _backendIndex(n), alpha, a, _backendIndex(lda), b, _backendIndex(ldb), beta, c, _backendIndex(ldc))
        #else
        _symm(layout: layout, side: side, triangle: triangle, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dsymm(layout: Layout, side: Side, triangle: Triangle, m: Int, n: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, b: UnsafePointer<Double>, ldb: Int, beta: Double, c: UnsafeMutablePointer<Double>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dsymm(layout._cblas, side._cblas, triangle._cblas, _backendIndex(m), _backendIndex(n), alpha, a, _backendIndex(lda), b, _backendIndex(ldb), beta, c, _backendIndex(ldc))
        #else
        _symm(layout: layout, side: side, triangle: triangle, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func csymm(layout: Layout, side: Side, triangle: Triangle, m: Int, n: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafePointer<Complex<Float>>, ldb: Int, beta: Complex<Float>, c: UnsafeMutablePointer<Complex<Float>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_csymm(layout._cblas, side._cblas, triangle._cblas, _backendIndex(m), _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(b), _backendIndex(ldb), _complexFloatPointer(beta), _complexFloatPointer(c), _backendIndex(ldc))
            }
        }
        #else
        _symm(layout: layout, side: side, triangle: triangle, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zsymm(layout: Layout, side: Side, triangle: Triangle, m: Int, n: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafePointer<Complex<Double>>, ldb: Int, beta: Complex<Double>, c: UnsafeMutablePointer<Complex<Double>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zsymm(layout._cblas, side._cblas, triangle._cblas, _backendIndex(m), _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(b), _backendIndex(ldb), _complexDoublePointer(beta), _complexDoublePointer(c), _backendIndex(ldc))
            }
        }
        #else
        _symm(layout: layout, side: side, triangle: triangle, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func chemm(layout: Layout, side: Side, triangle: Triangle, m: Int, n: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafePointer<Complex<Float>>, ldb: Int, beta: Complex<Float>, c: UnsafeMutablePointer<Complex<Float>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_chemm(layout._cblas, side._cblas, triangle._cblas, _backendIndex(m), _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(b), _backendIndex(ldb), _complexFloatPointer(beta), _complexFloatPointer(c), _backendIndex(ldc))
            }
        }
        #else
        _symm(layout: layout, side: side, triangle: triangle, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc, hermitian: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zhemm(layout: Layout, side: Side, triangle: Triangle, m: Int, n: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafePointer<Complex<Double>>, ldb: Int, beta: Complex<Double>, c: UnsafeMutablePointer<Complex<Double>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zhemm(layout._cblas, side._cblas, triangle._cblas, _backendIndex(m), _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(b), _backendIndex(ldb), _complexDoublePointer(beta), _complexDoublePointer(c), _backendIndex(ldc))
            }
        }
        #else
        _symm(layout: layout, side: side, triangle: triangle, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc, hermitian: true)
        #endif
    }
}

// MARK: - Rank-k updates

extension BLAS {
    @inlinable
    @inline(always)
    public static func ssyrk(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, beta: Float, c: UnsafeMutablePointer<Float>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ssyrk(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), alpha, a, _backendIndex(lda), beta, c, _backendIndex(ldc))
        #else
        _rankK(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: alpha, a: a, lda: lda, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dsyrk(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, beta: Double, c: UnsafeMutablePointer<Double>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dsyrk(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), alpha, a, _backendIndex(lda), beta, c, _backendIndex(ldc))
        #else
        _rankK(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: alpha, a: a, lda: lda, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func csyrk(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, beta: Complex<Float>, c: UnsafeMutablePointer<Complex<Float>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_csyrk(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(beta), _complexFloatPointer(c), _backendIndex(ldc))
            }
        }
        #else
        _rankK(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: alpha, a: a, lda: lda, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zsyrk(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, beta: Complex<Double>, c: UnsafeMutablePointer<Complex<Double>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zsyrk(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(beta), _complexDoublePointer(c), _backendIndex(ldc))
            }
        }
        #else
        _rankK(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: alpha, a: a, lda: lda, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cherk(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Float, a: UnsafePointer<Complex<Float>>, lda: Int, beta: Float, c: UnsafeMutablePointer<Complex<Float>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_cherk(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), alpha, _complexFloatPointer(a), _backendIndex(lda), beta, _complexFloatPointer(c), _backendIndex(ldc))
        #else
        _rankK(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: Complex<Float>(alpha, 0), a: a, lda: lda, beta: Complex<Float>(beta, 0), c: c, ldc: ldc, hermitian: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zherk(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Double, a: UnsafePointer<Complex<Double>>, lda: Int, beta: Double, c: UnsafeMutablePointer<Complex<Double>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_zherk(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), alpha, _complexDoublePointer(a), _backendIndex(lda), beta, _complexDoublePointer(c), _backendIndex(ldc))
        #else
        _rankK(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: Complex<Double>(alpha, 0), a: a, lda: lda, beta: Complex<Double>(beta, 0), c: c, ldc: ldc, hermitian: true)
        #endif
    }

    @inlinable
    @inline(always)
    public static func ssyr2k(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, b: UnsafePointer<Float>, ldb: Int, beta: Float, c: UnsafeMutablePointer<Float>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ssyr2k(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), alpha, a, _backendIndex(lda), b, _backendIndex(ldb), beta, c, _backendIndex(ldc))
        #else
        _rank2K(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dsyr2k(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, b: UnsafePointer<Double>, ldb: Int, beta: Double, c: UnsafeMutablePointer<Double>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dsyr2k(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), alpha, a, _backendIndex(lda), b, _backendIndex(ldb), beta, c, _backendIndex(ldc))
        #else
        _rank2K(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func csyr2k(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafePointer<Complex<Float>>, ldb: Int, beta: Complex<Float>, c: UnsafeMutablePointer<Complex<Float>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_csyr2k(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(b), _backendIndex(ldb), _complexFloatPointer(beta), _complexFloatPointer(c), _backendIndex(ldc))
            }
        }
        #else
        _rank2K(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zsyr2k(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafePointer<Complex<Double>>, ldb: Int, beta: Complex<Double>, c: UnsafeMutablePointer<Complex<Double>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zsyr2k(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(b), _backendIndex(ldb), _complexDoublePointer(beta), _complexDoublePointer(c), _backendIndex(ldc))
            }
        }
        #else
        _rank2K(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cher2k(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafePointer<Complex<Float>>, ldb: Int, beta: Float, c: UnsafeMutablePointer<Complex<Float>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_cher2k(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(b), _backendIndex(ldb), beta, _complexFloatPointer(c), _backendIndex(ldc))
        }
        #else
        _rank2K(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: Complex<Float>(beta, 0), c: c, ldc: ldc, hermitian: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zher2k(layout: Layout, triangle: Triangle, transpose: Transpose, n: Int, k: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafePointer<Complex<Double>>, ldb: Int, beta: Double, c: UnsafeMutablePointer<Complex<Double>>, ldc: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_zher2k(layout._cblas, triangle._cblas, transpose._cblas, _backendIndex(n), _backendIndex(k), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(b), _backendIndex(ldb), beta, _complexDoublePointer(c), _backendIndex(ldc))
        }
        #else
        _rank2K(layout: layout, triangle: triangle, transpose: transpose, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: Complex<Double>(beta, 0), c: c, ldc: ldc, hermitian: true)
        #endif
    }
}

extension BLAS {
    @inlinable
    @inline(always)
    public static func sgemmt(layout: Layout, triangle: Triangle, transposeA: Transpose, transposeB: Transpose, m: Int, k: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, b: UnsafePointer<Float>, ldb: Int, beta: Float, c: UnsafeMutablePointer<Float>, ldc: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sgemmt(layout._cblas, triangle._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(k), alpha, a, _backendIndex(lda), b, _backendIndex(ldb), beta, c, _backendIndex(ldc))
        #else
        _gemmt(layout: layout, triangle: triangle, transposeA: transposeA, transposeB: transposeB, m: m, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dgemmt(layout: Layout, triangle: Triangle, transposeA: Transpose, transposeB: Transpose, m: Int, k: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, b: UnsafePointer<Double>, ldb: Int, beta: Double, c: UnsafeMutablePointer<Double>, ldc: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dgemmt(layout._cblas, triangle._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(k), alpha, a, _backendIndex(lda), b, _backendIndex(ldb), beta, c, _backendIndex(ldc))
        #else
        _gemmt(layout: layout, triangle: triangle, transposeA: transposeA, transposeB: transposeB, m: m, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cgemmt(layout: Layout, triangle: Triangle, transposeA: Transpose, transposeB: Transpose, m: Int, k: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafePointer<Complex<Float>>, ldb: Int, beta: Complex<Float>, c: UnsafeMutablePointer<Complex<Float>>, ldc: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        var alpha = alpha; var beta = beta
        cblas_cgemmt(layout._cblas, triangle._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(k), &alpha, _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(b), _backendIndex(ldb), &beta, _complexFloatPointer(c), _backendIndex(ldc))
        #else
        _gemmt(layout: layout, triangle: triangle, transposeA: transposeA, transposeB: transposeB, m: m, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zgemmt(layout: Layout, triangle: Triangle, transposeA: Transpose, transposeB: Transpose, m: Int, k: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafePointer<Complex<Double>>, ldb: Int, beta: Complex<Double>, c: UnsafeMutablePointer<Complex<Double>>, ldc: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        var alpha = alpha; var beta = beta
        cblas_zgemmt(layout._cblas, triangle._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(k), &alpha, _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(b), _backendIndex(ldb), &beta, _complexDoublePointer(c), _backendIndex(ldc))
        #else
        _gemmt(layout: layout, triangle: triangle, transposeA: transposeA, transposeB: transposeB, m: m, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc)
        #endif
    }
}

// MARK: - Triangular matrix operations

extension BLAS {
    @inlinable
    @inline(always)
    public static func strmm(layout: Layout, side: Side, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, m: Int, n: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, b: UnsafeMutablePointer<Float>, ldb: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_strmm(layout._cblas, side._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(m), _backendIndex(n), alpha, a, _backendIndex(lda), b, _backendIndex(ldb))
        #else
        _trmm(layout: layout, side: side, triangle: triangle, transpose: transpose, diagonal: diagonal, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dtrmm(layout: Layout, side: Side, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, m: Int, n: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, b: UnsafeMutablePointer<Double>, ldb: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dtrmm(layout._cblas, side._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(m), _backendIndex(n), alpha, a, _backendIndex(lda), b, _backendIndex(ldb))
        #else
        _trmm(layout: layout, side: side, triangle: triangle, transpose: transpose, diagonal: diagonal, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ctrmm(layout: Layout, side: Side, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, m: Int, n: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafeMutablePointer<Complex<Float>>, ldb: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_ctrmm(layout._cblas, side._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(m), _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(b), _backendIndex(ldb))
        }
        #else
        _trmm(layout: layout, side: side, triangle: triangle, transpose: transpose, diagonal: diagonal, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ztrmm(layout: Layout, side: Side, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, m: Int, n: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafeMutablePointer<Complex<Double>>, ldb: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_ztrmm(layout._cblas, side._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(m), _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(b), _backendIndex(ldb))
        }
        #else
        _trmm(layout: layout, side: side, triangle: triangle, transpose: transpose, diagonal: diagonal, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }

    @inlinable
    @inline(always)
    public static func strsm(layout: Layout, side: Side, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, m: Int, n: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, b: UnsafeMutablePointer<Float>, ldb: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_strsm(layout._cblas, side._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(m), _backendIndex(n), alpha, a, _backendIndex(lda), b, _backendIndex(ldb))
        #else
        _trsm(layout: layout, side: side, triangle: triangle, transpose: transpose, diagonal: diagonal, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dtrsm(layout: Layout, side: Side, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, m: Int, n: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, b: UnsafeMutablePointer<Double>, ldb: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dtrsm(layout._cblas, side._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(m), _backendIndex(n), alpha, a, _backendIndex(lda), b, _backendIndex(ldb))
        #else
        _trsm(layout: layout, side: side, triangle: triangle, transpose: transpose, diagonal: diagonal, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ctrsm(layout: Layout, side: Side, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, m: Int, n: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafeMutablePointer<Complex<Float>>, ldb: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_ctrsm(layout._cblas, side._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(m), _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(b), _backendIndex(ldb))
        }
        #else
        _trsm(layout: layout, side: side, triangle: triangle, transpose: transpose, diagonal: diagonal, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ztrsm(layout: Layout, side: Side, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, m: Int, n: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafeMutablePointer<Complex<Double>>, ldb: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_ztrsm(layout._cblas, side._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(m), _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(b), _backendIndex(ldb))
        }
        #else
        _trsm(layout: layout, side: side, triangle: triangle, transpose: transpose, diagonal: diagonal, m: m, n: n, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }
}
