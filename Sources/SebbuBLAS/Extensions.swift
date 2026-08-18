// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

#if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
import COpenBLAS
#endif

import ComplexModule
import RealModule

extension BLAS {
    /// Reports an invalid BLAS parameter in the portable implementation.
    ///
    /// CBLAS declares this routine with C varargs, which Swift cannot expose
    /// safely. The fixed portion of the interface is retained and traps.
    public static func xerbla(
        parameter: Int,
        routine: UnsafePointer<CChar>,
        format: UnsafePointer<CChar>
    ) -> Never {
        _ = format
        var routineNameLength = 0
        while routine[routineNameLength] != 0 {
            routineNameLength += 1
        }
        _invokeXerblaHandler(
            routineName: UnsafeBufferPointer(
                start: routine,
                count: routineNameLength
            ),
            parameter: parameter
        )
        preconditionFailure("Invalid BLAS parameter at one-based position \(parameter)")
    }
}

extension BLAS {
    @inlinable
    @inline(always)
    public static func saxpby(n: Int, alpha: Float, x: UnsafePointer<Float>, incX: Int, beta: Float, y: UnsafeMutablePointer<Float>, incY: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_saxpby(_backendIndex(n), alpha, x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _axpby(n, alpha: alpha, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func daxpby(n: Int, alpha: Double, x: UnsafePointer<Double>, incX: Int, beta: Double, y: UnsafeMutablePointer<Double>, incY: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_daxpby(_backendIndex(n), alpha, x, _backendIndex(incX), beta, y, _backendIndex(incY))
        #else
        _axpby(n, alpha: alpha, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func caxpby(n: Int, alpha: Complex<Float>, x: UnsafePointer<Complex<Float>>, incX: Int, beta: Complex<Float>, y: UnsafeMutablePointer<Complex<Float>>, incY: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_caxpby(_backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(beta), _complexFloatPointer(y), _backendIndex(incY))
            }
        }
        #else
        _axpby(n, alpha: alpha, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zaxpby(n: Int, alpha: Complex<Double>, x: UnsafePointer<Complex<Double>>, incX: Int, beta: Complex<Double>, y: UnsafeMutablePointer<Complex<Double>>, incY: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            withUnsafePointer(to: beta) { beta in
                cblas_zaxpby(_backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(beta), _complexDoublePointer(y), _backendIndex(incY))
            }
        }
        #else
        _axpby(n, alpha: alpha, x: x, incX: incX, beta: beta, y: y, incY: incY)
        #endif
    }
}

extension BLAS {
    @inlinable
    @inline(always)
    public static func somatcopy(layout: Layout, transpose: Transpose, rows: Int, columns: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, b: UnsafeMutablePointer<Float>, ldb: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_somatcopy(layout._cblas, transpose._cblas, _backendIndex(rows), _backendIndex(columns), alpha, a, _backendIndex(lda), b, _backendIndex(ldb))
        #else
        _omatcopy(layout: layout, transpose: transpose, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func domatcopy(layout: Layout, transpose: Transpose, rows: Int, columns: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, b: UnsafeMutablePointer<Double>, ldb: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_domatcopy(layout._cblas, transpose._cblas, _backendIndex(rows), _backendIndex(columns), alpha, a, _backendIndex(lda), b, _backendIndex(ldb))
        #else
        _omatcopy(layout: layout, transpose: transpose, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func comatcopy(layout: Layout, transpose: Transpose, rows: Int, columns: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafeMutablePointer<Complex<Float>>, ldb: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alphaPointer in
            cblas_comatcopy(layout._cblas, transpose._cblas, _backendIndex(rows), _backendIndex(columns), UnsafeRawPointer(alphaPointer).assumingMemoryBound(to: Float.self), _complexFloatPointer(a).assumingMemoryBound(to: Float.self), _backendIndex(lda), _complexFloatPointer(b).assumingMemoryBound(to: Float.self), _backendIndex(ldb))
        }
        #else
        _omatcopy(layout: layout, transpose: transpose, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zomatcopy(layout: Layout, transpose: Transpose, rows: Int, columns: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafeMutablePointer<Complex<Double>>, ldb: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alphaPointer in
            cblas_zomatcopy(layout._cblas, transpose._cblas, _backendIndex(rows), _backendIndex(columns), UnsafeRawPointer(alphaPointer).assumingMemoryBound(to: Double.self), _complexDoublePointer(a).assumingMemoryBound(to: Double.self), _backendIndex(lda), _complexDoublePointer(b).assumingMemoryBound(to: Double.self), _backendIndex(ldb))
        }
        #else
        _omatcopy(layout: layout, transpose: transpose, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }

    @inlinable
    @inline(always)
    public static func simatcopy(layout: Layout, transpose: Transpose, rows: Int, columns: Int, alpha: Float, a: UnsafeMutablePointer<Float>, lda: Int, ldb: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_simatcopy(layout._cblas, transpose._cblas, _backendIndex(rows), _backendIndex(columns), alpha, a, _backendIndex(lda), _backendIndex(ldb))
        #else
        _imatcopy(layout: layout, transpose: transpose, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dimatcopy(layout: Layout, transpose: Transpose, rows: Int, columns: Int, alpha: Double, a: UnsafeMutablePointer<Double>, lda: Int, ldb: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dimatcopy(layout._cblas, transpose._cblas, _backendIndex(rows), _backendIndex(columns), alpha, a, _backendIndex(lda), _backendIndex(ldb))
        #else
        _imatcopy(layout: layout, transpose: transpose, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cimatcopy(layout: Layout, transpose: Transpose, rows: Int, columns: Int, alpha: Complex<Float>, a: UnsafeMutablePointer<Complex<Float>>, lda: Int, ldb: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alphaPointer in
            cblas_cimatcopy(layout._cblas, transpose._cblas, _backendIndex(rows), _backendIndex(columns), UnsafeRawPointer(alphaPointer).assumingMemoryBound(to: Float.self), _complexFloatPointer(a).assumingMemoryBound(to: Float.self), _backendIndex(lda), _backendIndex(ldb))
        }
        #else
        _imatcopy(layout: layout, transpose: transpose, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, ldb: ldb)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zimatcopy(layout: Layout, transpose: Transpose, rows: Int, columns: Int, alpha: Complex<Double>, a: UnsafeMutablePointer<Complex<Double>>, lda: Int, ldb: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alphaPointer in
            cblas_zimatcopy(layout._cblas, transpose._cblas, _backendIndex(rows), _backendIndex(columns), UnsafeRawPointer(alphaPointer).assumingMemoryBound(to: Double.self), _complexDoublePointer(a).assumingMemoryBound(to: Double.self), _backendIndex(lda), _backendIndex(ldb))
        }
        #else
        _imatcopy(layout: layout, transpose: transpose, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, ldb: ldb)
        #endif
    }
}

extension BLAS {
    @inlinable
    @inline(always)
    public static func sgeadd(layout: Layout, transposeA: Transpose, transposeC: Transpose, rows: Int, columns: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, beta: Float, c: UnsafeMutablePointer<Float>, ldc: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sgeadd(layout._cblas, transposeA._cblas, transposeC._cblas, _backendIndex(rows), _backendIndex(columns), alpha, a, _backendIndex(lda), beta, c, _backendIndex(ldc))
        #else
        _geadd(layout: layout, transposeA: transposeA, transposeC: transposeC, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, beta: beta, c: c, ldc: ldc)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dgeadd(layout: Layout, transposeA: Transpose, transposeC: Transpose, rows: Int, columns: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, beta: Double, c: UnsafeMutablePointer<Double>, ldc: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dgeadd(layout._cblas, transposeA._cblas, transposeC._cblas, _backendIndex(rows), _backendIndex(columns), alpha, a, _backendIndex(lda), beta, c, _backendIndex(ldc))
        #else
        _geadd(layout: layout, transposeA: transposeA, transposeC: transposeC, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, beta: beta, c: c, ldc: ldc)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cgeadd(layout: Layout, transposeA: Transpose, transposeC: Transpose, rows: Int, columns: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, beta: Complex<Float>, c: UnsafeMutablePointer<Complex<Float>>, ldc: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        var alpha = alpha; var beta = beta
        withUnsafePointer(to: &alpha) { alphaPointer in
            withUnsafePointer(to: &beta) { betaPointer in
                cblas_cgeadd(layout._cblas, transposeA._cblas, transposeC._cblas, _backendIndex(rows), _backendIndex(columns), UnsafeRawPointer(alphaPointer).assumingMemoryBound(to: Float.self), _complexFloatPointer(a).assumingMemoryBound(to: Float.self), _backendIndex(lda), UnsafeRawPointer(betaPointer).assumingMemoryBound(to: Float.self), _complexFloatPointer(c).assumingMemoryBound(to: Float.self), _backendIndex(ldc))
            }
        }
        #else
        _geadd(layout: layout, transposeA: transposeA, transposeC: transposeC, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, beta: beta, c: c, ldc: ldc)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zgeadd(layout: Layout, transposeA: Transpose, transposeC: Transpose, rows: Int, columns: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, beta: Complex<Double>, c: UnsafeMutablePointer<Complex<Double>>, ldc: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        var alpha = alpha; var beta = beta
        withUnsafePointer(to: &alpha) { alphaPointer in
            withUnsafePointer(to: &beta) { betaPointer in
                cblas_zgeadd(layout._cblas, transposeA._cblas, transposeC._cblas, _backendIndex(rows), _backendIndex(columns), UnsafeRawPointer(alphaPointer).assumingMemoryBound(to: Double.self), _complexDoublePointer(a).assumingMemoryBound(to: Double.self), _backendIndex(lda), UnsafeRawPointer(betaPointer).assumingMemoryBound(to: Double.self), _complexDoublePointer(c).assumingMemoryBound(to: Double.self), _backendIndex(ldc))
            }
        }
        #else
        _geadd(layout: layout, transposeA: transposeA, transposeC: transposeC, rows: rows, columns: columns, alpha: alpha, a: a, lda: lda, beta: beta, c: c, ldc: ldc)
        #endif
    }
}
