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
    public static func sgemv(
        layout: Layout, transpose: Transpose, m: Int, n: Int,
        alpha: Float, a: UnsafePointer<Float>, lda: Int,
        x: UnsafePointer<Float>, incX: Int,
        beta: Float, y: UnsafeMutablePointer<Float>, incY: Int
    ) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sgemv(layout._cblas, transpose._cblas, _backendIndex(m), _backendIndex(n), alpha, a, _backendIndex(lda), x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _gemv(layout: layout, transpose: transpose, rows: m, columns: n, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func dgemv(
        layout: Layout, transpose: Transpose, m: Int, n: Int,
        alpha: Double, a: UnsafePointer<Double>, lda: Int,
        x: UnsafePointer<Double>, incX: Int,
        beta: Double, y: UnsafeMutablePointer<Double>, incY: Int
    ) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dgemv(layout._cblas, transpose._cblas, _backendIndex(m), _backendIndex(n), alpha, a, _backendIndex(lda), x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _gemv(layout: layout, transpose: transpose, rows: m, columns: n, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func cgemv(
        layout: Layout, transpose: Transpose, m: Int, n: Int,
        alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int,
        x: UnsafePointer<Complex<Float>>, incX: Int,
        beta: Complex<Float>, y: UnsafeMutablePointer<Complex<Float>>, incY: Int
    ) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_cgemv(layout._cblas, transpose._cblas, _backendIndex(m), _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(beta), _complexFloatPointer(y), _backendIndex(incY))
            }
        }
        #else
        _gemv(layout: layout, transpose: transpose, rows: m, columns: n, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func zgemv(
        layout: Layout, transpose: Transpose, m: Int, n: Int,
        alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int,
        x: UnsafePointer<Complex<Double>>, incX: Int,
        beta: Complex<Double>, y: UnsafeMutablePointer<Complex<Double>>, incY: Int
    ) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zgemv(layout._cblas, transpose._cblas, _backendIndex(m), _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(beta), _complexDoublePointer(y), _backendIndex(incY))
            }
        }
        #else
        _gemv(layout: layout, transpose: transpose, rows: m, columns: n, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }
}

// MARK: - General and symmetric band matrices

extension BLAS {
    @inlinable
    @inline(always)
    public static func sgbmv(layout: Layout, transpose: Transpose, m: Int, n: Int, kl: Int, ku: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, x: UnsafePointer<Float>, incX: Int, beta: Float, y: UnsafeMutablePointer<Float>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sgbmv(layout._cblas, transpose._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(kl), _backendIndex(ku), alpha, a, _backendIndex(lda), x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _gbmv(layout: layout, transpose: transpose, rows: m, columns: n, lowerBandwidth: kl, upperBandwidth: ku, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dgbmv(layout: Layout, transpose: Transpose, m: Int, n: Int, kl: Int, ku: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, x: UnsafePointer<Double>, incX: Int, beta: Double, y: UnsafeMutablePointer<Double>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dgbmv(layout._cblas, transpose._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(kl), _backendIndex(ku), alpha, a, _backendIndex(lda), x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _gbmv(layout: layout, transpose: transpose, rows: m, columns: n, lowerBandwidth: kl, upperBandwidth: ku, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cgbmv(layout: Layout, transpose: Transpose, m: Int, n: Int, kl: Int, ku: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, x: UnsafePointer<Complex<Float>>, incX: Int, beta: Complex<Float>, y: UnsafeMutablePointer<Complex<Float>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_cgbmv(layout._cblas, transpose._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(kl), _backendIndex(ku), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(beta), _complexFloatPointer(y), _backendIndex(incY))
            }
        }
        #else
        _gbmv(layout: layout, transpose: transpose, rows: m, columns: n, lowerBandwidth: kl, upperBandwidth: ku, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zgbmv(layout: Layout, transpose: Transpose, m: Int, n: Int, kl: Int, ku: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, x: UnsafePointer<Complex<Double>>, incX: Int, beta: Complex<Double>, y: UnsafeMutablePointer<Complex<Double>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zgbmv(layout._cblas, transpose._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(kl), _backendIndex(ku), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(beta), _complexDoublePointer(y), _backendIndex(incY))
            }
        }
        #else
        _gbmv(layout: layout, transpose: transpose, rows: m, columns: n, lowerBandwidth: kl, upperBandwidth: ku, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func ssbmv(layout: Layout, triangle: Triangle, n: Int, k: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, x: UnsafePointer<Float>, incX: Int, beta: Float, y: UnsafeMutablePointer<Float>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ssbmv(layout._cblas, triangle._cblas, _backendIndex(n), _backendIndex(k), alpha, a, _backendIndex(lda), x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _sbmv(layout: layout, triangle: triangle, order: n, bandwidth: k, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dsbmv(layout: Layout, triangle: Triangle, n: Int, k: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, x: UnsafePointer<Double>, incX: Int, beta: Double, y: UnsafeMutablePointer<Double>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dsbmv(layout._cblas, triangle._cblas, _backendIndex(n), _backendIndex(k), alpha, a, _backendIndex(lda), x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _sbmv(layout: layout, triangle: triangle, order: n, bandwidth: k, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func chbmv(layout: Layout, triangle: Triangle, n: Int, k: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, x: UnsafePointer<Complex<Float>>, incX: Int, beta: Complex<Float>, y: UnsafeMutablePointer<Complex<Float>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_chbmv(layout._cblas, triangle._cblas, _backendIndex(n), _backendIndex(k), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(beta), _complexFloatPointer(y), _backendIndex(incY))
            }
        }
        #else
        _sbmv(layout: layout, triangle: triangle, order: n, bandwidth: k, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zhbmv(layout: Layout, triangle: Triangle, n: Int, k: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, x: UnsafePointer<Complex<Double>>, incX: Int, beta: Complex<Double>, y: UnsafeMutablePointer<Complex<Double>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zhbmv(layout._cblas, triangle._cblas, _backendIndex(n), _backendIndex(k), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(beta), _complexDoublePointer(y), _backendIndex(incY))
            }
        }
        #else
        _sbmv(layout: layout, triangle: triangle, order: n, bandwidth: k, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: true)
        #endif
    }
}

// MARK: - Triangular band matrices

extension BLAS {
    @inlinable
    @inline(always)
    public static func stbmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, k: Int, a: UnsafePointer<Float>, lda: Int, x: UnsafeMutablePointer<Float>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_stbmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _backendIndex(k), a, _backendIndex(lda), x, _backendIndex(incX))
        #else
        _tbmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, bandwidth: k, matrix: a, leadingDimension: lda, x: x, incX: incX, solve: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dtbmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, k: Int, a: UnsafePointer<Double>, lda: Int, x: UnsafeMutablePointer<Double>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dtbmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _backendIndex(k), a, _backendIndex(lda), x, _backendIndex(incX))
        #else
        _tbmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, bandwidth: k, matrix: a, leadingDimension: lda, x: x, incX: incX, solve: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ctbmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, k: Int, a: UnsafePointer<Complex<Float>>, lda: Int, x: UnsafeMutablePointer<Complex<Float>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ctbmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _backendIndex(k), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(x), _backendIndex(incX))
        #else
        _tbmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, bandwidth: k, matrix: a, leadingDimension: lda, x: x, incX: incX, solve: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ztbmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, k: Int, a: UnsafePointer<Complex<Double>>, lda: Int, x: UnsafeMutablePointer<Complex<Double>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ztbmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _backendIndex(k), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _tbmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, bandwidth: k, matrix: a, leadingDimension: lda, x: x, incX: incX, solve: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func stbsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, k: Int, a: UnsafePointer<Float>, lda: Int, x: UnsafeMutablePointer<Float>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_stbsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _backendIndex(k), a, _backendIndex(lda), x, _backendIndex(incX))
        #else
        _tbmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, bandwidth: k, matrix: a, leadingDimension: lda, x: x, incX: incX, solve: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dtbsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, k: Int, a: UnsafePointer<Double>, lda: Int, x: UnsafeMutablePointer<Double>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dtbsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _backendIndex(k), a, _backendIndex(lda), x, _backendIndex(incX))
        #else
        _tbmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, bandwidth: k, matrix: a, leadingDimension: lda, x: x, incX: incX, solve: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ctbsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, k: Int, a: UnsafePointer<Complex<Float>>, lda: Int, x: UnsafeMutablePointer<Complex<Float>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ctbsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _backendIndex(k), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(x), _backendIndex(incX))
        #else
        _tbmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, bandwidth: k, matrix: a, leadingDimension: lda, x: x, incX: incX, solve: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ztbsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, k: Int, a: UnsafePointer<Complex<Double>>, lda: Int, x: UnsafeMutablePointer<Complex<Double>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ztbsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _backendIndex(k), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _tbmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, bandwidth: k, matrix: a, leadingDimension: lda, x: x, incX: incX, solve: true)
        #endif
    }
}

// MARK: - General rank-one updates

extension BLAS {
    @inlinable
    @inline(always)
    public static func sger(layout: Layout, m: Int, n: Int, alpha: Float, x: UnsafePointer<Float>, incX: Int, y: UnsafePointer<Float>, incY: Int, a: UnsafeMutablePointer<Float>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sger(layout._cblas, _backendIndex(m), _backendIndex(n), alpha, x, _backendIndex(incX), y, _backendIndex(incY), a, _backendIndex(lda))
        #else
        _ger(layout: layout, rows: m, columns: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, matrix: a, leadingDimension: lda, conjugateY: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dger(layout: Layout, m: Int, n: Int, alpha: Double, x: UnsafePointer<Double>, incX: Int, y: UnsafePointer<Double>, incY: Int, a: UnsafeMutablePointer<Double>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dger(layout._cblas, _backendIndex(m), _backendIndex(n), alpha, x, _backendIndex(incX), y, _backendIndex(incY), a, _backendIndex(lda))
        #else
        _ger(layout: layout, rows: m, columns: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, matrix: a, leadingDimension: lda, conjugateY: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cgeru(layout: Layout, m: Int, n: Int, alpha: Complex<Float>, x: UnsafePointer<Complex<Float>>, incX: Int, y: UnsafePointer<Complex<Float>>, incY: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_cgeru(layout._cblas, _backendIndex(m), _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY), _complexFloatPointer(a), _backendIndex(lda))
        }
        #else
        _ger(layout: layout, rows: m, columns: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, matrix: a, leadingDimension: lda, conjugateY: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cgerc(layout: Layout, m: Int, n: Int, alpha: Complex<Float>, x: UnsafePointer<Complex<Float>>, incX: Int, y: UnsafePointer<Complex<Float>>, incY: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_cgerc(layout._cblas, _backendIndex(m), _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY), _complexFloatPointer(a), _backendIndex(lda))
        }
        #else
        _ger(layout: layout, rows: m, columns: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, matrix: a, leadingDimension: lda, conjugateY: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zgeru(layout: Layout, m: Int, n: Int, alpha: Complex<Double>, x: UnsafePointer<Complex<Double>>, incX: Int, y: UnsafePointer<Complex<Double>>, incY: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_zgeru(layout._cblas, _backendIndex(m), _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY), _complexDoublePointer(a), _backendIndex(lda))
        }
        
        #else
        _ger(layout: layout, rows: m, columns: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, matrix: a, leadingDimension: lda, conjugateY: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zgerc(layout: Layout, m: Int, n: Int, alpha: Complex<Double>, x: UnsafePointer<Complex<Double>>, incX: Int, y: UnsafePointer<Complex<Double>>, incY: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_zgerc(layout._cblas, _backendIndex(m), _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY), _complexDoublePointer(a), _backendIndex(lda))
        }
        #else
        _ger(layout: layout, rows: m, columns: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, matrix: a, leadingDimension: lda, conjugateY: true)
        #endif
    }
}

// MARK: - Dense triangular matrices

extension BLAS {
    @inlinable
    @inline(always)
    public static func strmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, a: UnsafePointer<Float>, lda: Int, x: UnsafeMutablePointer<Float>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_strmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), a, _backendIndex(lda), x, _backendIndex(incX))
        #else
        _trmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, matrix: a, leadingDimension: lda, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dtrmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, a: UnsafePointer<Double>, lda: Int, x: UnsafeMutablePointer<Double>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dtrmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), a, _backendIndex(lda), x, _backendIndex(incX))
        #else
        _trmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, matrix: a, leadingDimension: lda, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ctrmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, a: UnsafePointer<Complex<Float>>, lda: Int, x: UnsafeMutablePointer<Complex<Float>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ctrmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(x), _backendIndex(incX))
        #else
        _trmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, matrix: a, leadingDimension: lda, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ztrmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, a: UnsafePointer<Complex<Double>>, lda: Int, x: UnsafeMutablePointer<Complex<Double>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ztrmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _trmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, matrix: a, leadingDimension: lda, x: x, incX: incX)
        #endif
    }

    @inlinable
    @inline(always)
    public static func strsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, a: UnsafePointer<Float>, lda: Int, x: UnsafeMutablePointer<Float>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_strsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), a, _backendIndex(lda), x, _backendIndex(incX))
        #else
        _trsv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, matrix: a, leadingDimension: lda, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dtrsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, a: UnsafePointer<Double>, lda: Int, x: UnsafeMutablePointer<Double>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dtrsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), a, _backendIndex(lda), x, _backendIndex(incX))
        #else
        _trsv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, matrix: a, leadingDimension: lda, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ctrsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, a: UnsafePointer<Complex<Float>>, lda: Int, x: UnsafeMutablePointer<Complex<Float>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ctrsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(x), _backendIndex(incX))
        #else
        _trsv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, matrix: a, leadingDimension: lda, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ztrsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, a: UnsafePointer<Complex<Double>>, lda: Int, x: UnsafeMutablePointer<Complex<Double>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ztrsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _trsv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, matrix: a, leadingDimension: lda, x: x, incX: incX)
        #endif
    }
}

// MARK: - Symmetric and Hermitian rank updates

extension BLAS {
    @inlinable
    @inline(always)
    public static func ssyr(layout: Layout, triangle: Triangle, n: Int, alpha: Float, x: UnsafePointer<Float>, incX: Int, a: UnsafeMutablePointer<Float>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ssyr(layout._cblas, triangle._cblas, _backendIndex(n), alpha, x, _backendIndex(incX), a, _backendIndex(lda))
        #else
        _rank1Triangle(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, matrix: a, leadingDimension: lda, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dsyr(layout: Layout, triangle: Triangle, n: Int, alpha: Double, x: UnsafePointer<Double>, incX: Int, a: UnsafeMutablePointer<Double>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dsyr(layout._cblas, triangle._cblas, _backendIndex(n), alpha, x, _backendIndex(incX), a, _backendIndex(lda))
        #else
        _rank1Triangle(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, matrix: a, leadingDimension: lda, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cher(layout: Layout, triangle: Triangle, n: Int, alpha: Float, x: UnsafePointer<Complex<Float>>, incX: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_cher(layout._cblas, triangle._cblas, _backendIndex(n), alpha, _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(a), _backendIndex(lda))
        #else
        _rank1Triangle(layout: layout, triangle: triangle, order: n, alpha: Complex<Float>(alpha, 0), x: x, incX: incX, matrix: a, leadingDimension: lda, hermitian: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zher(layout: Layout, triangle: Triangle, n: Int, alpha: Double, x: UnsafePointer<Complex<Double>>, incX: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_zher(layout._cblas, triangle._cblas, _backendIndex(n), alpha, _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(a), _backendIndex(lda))
        #else
        _rank1Triangle(layout: layout, triangle: triangle, order: n, alpha: Complex<Double>(alpha, 0), x: x, incX: incX, matrix: a, leadingDimension: lda, hermitian: true)
        #endif
    }

    @inlinable
    @inline(always)
    public static func ssyr2(layout: Layout, triangle: Triangle, n: Int, alpha: Float, x: UnsafePointer<Float>, incX: Int, y: UnsafePointer<Float>, incY: Int, a: UnsafeMutablePointer<Float>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ssyr2(layout._cblas, triangle._cblas, _backendIndex(n), alpha, x, _backendIndex(incX), y, _backendIndex(incY), a, _backendIndex(lda))
        #else
        _rank2Triangle(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, matrix: a, leadingDimension: lda, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dsyr2(layout: Layout, triangle: Triangle, n: Int, alpha: Double, x: UnsafePointer<Double>, incX: Int, y: UnsafePointer<Double>, incY: Int, a: UnsafeMutablePointer<Double>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dsyr2(layout._cblas, triangle._cblas, _backendIndex(n), alpha, x, _backendIndex(incX), y, _backendIndex(incY), a, _backendIndex(lda))
        #else
        _rank2Triangle(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, matrix: a, leadingDimension: lda, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cher2(layout: Layout, triangle: Triangle, n: Int, alpha: Complex<Float>, x: UnsafePointer<Complex<Float>>, incX: Int, y: UnsafePointer<Complex<Float>>, incY: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_cher2(layout._cblas, triangle._cblas, _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY), _complexFloatPointer(a), _backendIndex(lda))
        }
        #else
        _rank2Triangle(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, matrix: a, leadingDimension: lda, hermitian: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zher2(layout: Layout, triangle: Triangle, n: Int, alpha: Complex<Double>, x: UnsafePointer<Complex<Double>>, incX: Int, y: UnsafePointer<Complex<Double>>, incY: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_zher2(layout._cblas, triangle._cblas, _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY), _complexDoublePointer(a), _backendIndex(lda))
        }
        #else
        _rank2Triangle(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, matrix: a, leadingDimension: lda, hermitian: true)
        #endif
    }
}

// MARK: - Packed triangular matrices

extension BLAS {
    @inlinable
    @inline(always)
    public static func stpmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, ap: UnsafePointer<Float>, x: UnsafeMutablePointer<Float>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_stpmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), ap, x, _backendIndex(incX))
        #else
        _tpmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, packedMatrix: ap, x: x, incX: incX, solve: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dtpmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, ap: UnsafePointer<Double>, x: UnsafeMutablePointer<Double>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dtpmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), ap, x, _backendIndex(incX))
        #else
        _tpmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, packedMatrix: ap, x: x, incX: incX, solve: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ctpmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, ap: UnsafePointer<Complex<Float>>, x: UnsafeMutablePointer<Complex<Float>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ctpmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _complexFloatPointer(ap), _complexFloatPointer(x), _backendIndex(incX))
        #else
        _tpmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, packedMatrix: ap, x: x, incX: incX, solve: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ztpmv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, ap: UnsafePointer<Complex<Double>>, x: UnsafeMutablePointer<Complex<Double>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ztpmv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _complexDoublePointer(ap), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _tpmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, packedMatrix: ap, x: x, incX: incX, solve: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func stpsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, ap: UnsafePointer<Float>, x: UnsafeMutablePointer<Float>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_stpsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), ap, x, _backendIndex(incX))
        #else
        _tpmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, packedMatrix: ap, x: x, incX: incX, solve: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dtpsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, ap: UnsafePointer<Double>, x: UnsafeMutablePointer<Double>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dtpsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), ap, x, _backendIndex(incX))
        #else
        _tpmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, packedMatrix: ap, x: x, incX: incX, solve: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ctpsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, ap: UnsafePointer<Complex<Float>>, x: UnsafeMutablePointer<Complex<Float>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ctpsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _complexFloatPointer(ap), _complexFloatPointer(x), _backendIndex(incX))
        #else
        _tpmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, packedMatrix: ap, x: x, incX: incX, solve: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ztpsv(layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int, ap: UnsafePointer<Complex<Double>>, x: UnsafeMutablePointer<Complex<Double>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ztpsv(layout._cblas, triangle._cblas, transpose._cblas, diagonal._cblas, _backendIndex(n), _complexDoublePointer(ap), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _tpmv(layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, order: n, packedMatrix: ap, x: x, incX: incX, solve: true)
        #endif
    }
}

// MARK: - Symmetric, Hermitian, and packed matrix-vector products

extension BLAS {
    @inlinable
    @inline(always)
    public static func ssymv(layout: Layout, triangle: Triangle, n: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, x: UnsafePointer<Float>, incX: Int, beta: Float, y: UnsafeMutablePointer<Float>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ssymv(layout._cblas, triangle._cblas, _backendIndex(n), alpha, a, _backendIndex(lda), x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _symv(layout: layout, triangle: triangle, order: n, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dsymv(layout: Layout, triangle: Triangle, n: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, x: UnsafePointer<Double>, incX: Int, beta: Double, y: UnsafeMutablePointer<Double>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dsymv(layout._cblas, triangle._cblas, _backendIndex(n), alpha, a, _backendIndex(lda), x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _symv(layout: layout, triangle: triangle, order: n, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func chemv(layout: Layout, triangle: Triangle, n: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, x: UnsafePointer<Complex<Float>>, incX: Int, beta: Complex<Float>, y: UnsafeMutablePointer<Complex<Float>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_chemv(layout._cblas, triangle._cblas, _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(beta), _complexFloatPointer(y), _backendIndex(incY))
            }
        }
        #else
        _symv(layout: layout, triangle: triangle, order: n, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zhemv(layout: Layout, triangle: Triangle, n: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, x: UnsafePointer<Complex<Double>>, incX: Int, beta: Complex<Double>, y: UnsafeMutablePointer<Complex<Double>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zhemv(layout._cblas, triangle._cblas, _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(beta), _complexDoublePointer(y), _backendIndex(incY))
            }
        }
        #else
        _symv(layout: layout, triangle: triangle, order: n, alpha: alpha, matrix: a, leadingDimension: lda, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: true)
        #endif
    }

    @inlinable
    @inline(always)
    public static func sspmv(layout: Layout, triangle: Triangle, n: Int, alpha: Float, ap: UnsafePointer<Float>, x: UnsafePointer<Float>, incX: Int, beta: Float, y: UnsafeMutablePointer<Float>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sspmv(layout._cblas, triangle._cblas, _backendIndex(n), alpha, ap, x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _spmv(layout: layout, triangle: triangle, order: n, alpha: alpha, packedMatrix: ap, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dspmv(layout: Layout, triangle: Triangle, n: Int, alpha: Double, ap: UnsafePointer<Double>, x: UnsafePointer<Double>, incX: Int, beta: Double, y: UnsafeMutablePointer<Double>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dspmv(layout._cblas, triangle._cblas, _backendIndex(n), alpha, ap, x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _spmv(layout: layout, triangle: triangle, order: n, alpha: alpha, packedMatrix: ap, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func chpmv(layout: Layout, triangle: Triangle, n: Int, alpha: Complex<Float>, ap: UnsafePointer<Complex<Float>>, x: UnsafePointer<Complex<Float>>, incX: Int, beta: Complex<Float>, y: UnsafeMutablePointer<Complex<Float>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_chpmv(layout._cblas, triangle._cblas, _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(ap), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(beta), _complexFloatPointer(y), _backendIndex(incY))
            }
        }
        #else
        _spmv(layout: layout, triangle: triangle, order: n, alpha: alpha, packedMatrix: ap, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zhpmv(layout: Layout, triangle: Triangle, n: Int, alpha: Complex<Double>, ap: UnsafePointer<Complex<Double>>, x: UnsafePointer<Complex<Double>>, incX: Int, beta: Complex<Double>, y: UnsafeMutablePointer<Complex<Double>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zhpmv(layout._cblas, triangle._cblas, _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(ap), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(beta), _complexDoublePointer(y), _backendIndex(incY))
            }
        }
        #else
        _spmv(layout: layout, triangle: triangle, order: n, alpha: alpha, packedMatrix: ap, x: x, incX: incX, beta: beta, y: y, incY: incY, hermitian: true)
        #endif
    }
}

// MARK: - Packed rank updates

extension BLAS {
    @inlinable
    @inline(always)
    public static func sspr(layout: Layout, triangle: Triangle, n: Int, alpha: Float, x: UnsafePointer<Float>, incX: Int, ap: UnsafeMutablePointer<Float>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sspr(layout._cblas, triangle._cblas, _backendIndex(n), alpha, x, _backendIndex(incX), ap)
        #else
        _packedRank1(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, packedMatrix: ap, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dspr(layout: Layout, triangle: Triangle, n: Int, alpha: Double, x: UnsafePointer<Double>, incX: Int, ap: UnsafeMutablePointer<Double>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dspr(layout._cblas, triangle._cblas, _backendIndex(n), alpha, x, _backendIndex(incX), ap)
        #else
        _packedRank1(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, packedMatrix: ap, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func chpr(layout: Layout, triangle: Triangle, n: Int, alpha: Float, x: UnsafePointer<Complex<Float>>, incX: Int, ap: UnsafeMutablePointer<Complex<Float>>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_chpr(layout._cblas, triangle._cblas, _backendIndex(n), alpha, _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(ap))
        #else
        _packedRank1(layout: layout, triangle: triangle, order: n, alpha: Complex<Float>(alpha, 0), x: x, incX: incX, packedMatrix: ap, hermitian: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zhpr(layout: Layout, triangle: Triangle, n: Int, alpha: Double, x: UnsafePointer<Complex<Double>>, incX: Int, ap: UnsafeMutablePointer<Complex<Double>>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_zhpr(layout._cblas, triangle._cblas, _backendIndex(n), alpha, _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(ap))
        #else
        _packedRank1(layout: layout, triangle: triangle, order: n, alpha: Complex<Double>(alpha, 0), x: x, incX: incX, packedMatrix: ap, hermitian: true)
        #endif
    }

    @inlinable
    @inline(always)
    public static func sspr2(layout: Layout, triangle: Triangle, n: Int, alpha: Float, x: UnsafePointer<Float>, incX: Int, y: UnsafePointer<Float>, incY: Int, ap: UnsafeMutablePointer<Float>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sspr2(layout._cblas, triangle._cblas, _backendIndex(n), alpha, x, _backendIndex(incX), y, _backendIndex(incY), ap)
        #else
        _packedRank2(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, packedMatrix: ap, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dspr2(layout: Layout, triangle: Triangle, n: Int, alpha: Double, x: UnsafePointer<Double>, incX: Int, y: UnsafePointer<Double>, incY: Int, ap: UnsafeMutablePointer<Double>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dspr2(layout._cblas, triangle._cblas, _backendIndex(n), alpha, x, _backendIndex(incX), y, _backendIndex(incY), ap)
        #else
        _packedRank2(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, packedMatrix: ap, hermitian: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func chpr2(layout: Layout, triangle: Triangle, n: Int, alpha: Complex<Float>, x: UnsafePointer<Complex<Float>>, incX: Int, y: UnsafePointer<Complex<Float>>, incY: Int, ap: UnsafeMutablePointer<Complex<Float>>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_chpr2(layout._cblas, triangle._cblas, _backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY), _complexFloatPointer(ap))
        }
        #else
        _packedRank2(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, packedMatrix: ap, hermitian: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zhpr2(layout: Layout, triangle: Triangle, n: Int, alpha: Complex<Double>, x: UnsafePointer<Complex<Double>>, incX: Int, y: UnsafePointer<Complex<Double>>, incY: Int, ap: UnsafeMutablePointer<Complex<Double>>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_zhpr2(layout._cblas, triangle._cblas, _backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY), _complexDoublePointer(ap))
        }
        #else
        _packedRank2(layout: layout, triangle: triangle, order: n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, packedMatrix: ap, hermitian: true)
        #endif
    }
}
