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
    public static func sdsdot(
        n: Int, alpha: Float,
        x: UnsafePointer<Float>, incX: Int,
        y: UnsafePointer<Float>, incY: Int
    ) -> Float {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        return cblas_sdsdot(_backendIndex(n), alpha, x, _backendIndex(incX), y, _backendIndex(incY))
        #else
        return Float(Double(alpha) + _mixedDot(n, x, incX, y, incY))
        #endif
    }

    @inlinable
    @inline(always)
    public static func dsdot(
        n: Int,
        x: UnsafePointer<Float>, incX: Int,
        y: UnsafePointer<Float>, incY: Int
    ) -> Double {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        return cblas_dsdot(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY))
        #else
        return _mixedDot(n, x, incX, y, incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func sdot(
        n: Int,
        x: UnsafePointer<Float>, incX: Int,
        y: UnsafePointer<Float>, incY: Int
    ) -> Float {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        return cblas_sdot(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY))
        #else
        return _dot(n, x, incX, y, incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func ddot(
        n: Int,
        x: UnsafePointer<Double>, incX: Int,
        y: UnsafePointer<Double>, incY: Int
    ) -> Double {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        return cblas_ddot(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY))
        #else
        return _dot(n, x, incX, y, incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func cdotu(
        n: Int,
        x: UnsafePointer<Complex<Float>>, incX: Int,
        y: UnsafePointer<Complex<Float>>, incY: Int
    ) -> Complex<Float> {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        var result = Complex<Float>.zero
        withUnsafeMutablePointer(to: &result) {
            cblas_cdotu_sub(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY), _complexFloatPointer($0))
        }
        return result
        #else
        return _dot(n, x, incX, y, incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func cdotc(
        n: Int,
        x: UnsafePointer<Complex<Float>>, incX: Int,
        y: UnsafePointer<Complex<Float>>, incY: Int
    ) -> Complex<Float> {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        var result = Complex<Float>.zero
        withUnsafeMutablePointer(to: &result) {
            cblas_cdotc_sub(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY), _complexFloatPointer($0))
        }
        return result
        #else
        return _dot(n, x, incX, y, incY, conjugateX: true)
        #endif
    }

    @inlinable
    @inline(always)
    public static func zdotu(
        n: Int,
        x: UnsafePointer<Complex<Double>>, incX: Int,
        y: UnsafePointer<Complex<Double>>, incY: Int
    ) -> Complex<Double> {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        var result = Complex<Double>.zero
        withUnsafeMutablePointer(to: &result) {
            cblas_zdotu_sub(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY), _complexDoublePointer($0))
        }
        return result
        #else
        return _dot(n, x, incX, y, incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func zdotc(
        n: Int,
        x: UnsafePointer<Complex<Double>>, incX: Int,
        y: UnsafePointer<Complex<Double>>, incY: Int
    ) -> Complex<Double> {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        var result = Complex<Double>.zero
        withUnsafeMutablePointer(to: &result) {
            cblas_zdotc_sub(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY), _complexDoublePointer($0))
        }
        return result
        #else
        return _dot(n, x, incX, y, incY, conjugateX: true)
        #endif
    }

    @inlinable
    @inline(always)
    public static func cdotuSub(
        n: Int,
        x: UnsafePointer<Complex<Float>>, incX: Int,
        y: UnsafePointer<Complex<Float>>, incY: Int,
        result: UnsafeMutablePointer<Complex<Float>>
    ) { result.pointee = cdotu(n: n, x: x, incX: incX, y: y, incY: incY) }

    @inlinable
    @inline(always)
    public static func cdotcSub(
        n: Int,
        x: UnsafePointer<Complex<Float>>, incX: Int,
        y: UnsafePointer<Complex<Float>>, incY: Int,
        result: UnsafeMutablePointer<Complex<Float>>
    ) { result.pointee = cdotc(n: n, x: x, incX: incX, y: y, incY: incY) }

    @inlinable
    @inline(always)
    public static func zdotuSub(
        n: Int,
        x: UnsafePointer<Complex<Double>>, incX: Int,
        y: UnsafePointer<Complex<Double>>, incY: Int,
        result: UnsafeMutablePointer<Complex<Double>>
    ) { result.pointee = zdotu(n: n, x: x, incX: incX, y: y, incY: incY) }

    @inlinable
    @inline(always)
    public static func zdotcSub(
        n: Int,
        x: UnsafePointer<Complex<Double>>, incX: Int,
        y: UnsafePointer<Complex<Double>>, incY: Int,
        result: UnsafeMutablePointer<Complex<Double>>
    ) { result.pointee = zdotc(n: n, x: x, incX: incX, y: y, incY: incY) }

    @inlinable
    @inline(always)
    public static func sasum(n: Int, x: UnsafePointer<Float>, incX: Int) -> Float {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sasum(_backendIndex(n), x, _backendIndex(incX))
        #else
        _asum(n, x, incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dasum(n: Int, x: UnsafePointer<Double>, incX: Int) -> Double {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dasum(_backendIndex(n), x, _backendIndex(incX))
        #else
        _asum(n, x, incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func scasum(n: Int, x: UnsafePointer<Complex<Float>>, incX: Int) -> Float {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_scasum(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX))
        #else
        _asum(n, x, incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dzasum(n: Int, x: UnsafePointer<Complex<Double>>, incX: Int) -> Double {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dzasum(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _asum(n, x, incX)
        #endif
    }

    @inlinable
    @inline(always)
    public static func ssum(n: Int, x: UnsafePointer<Float>, incX: Int) -> Float {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ssum(_backendIndex(n), x, _backendIndex(incX))
        #else
        _sum(n, x, incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dsum(n: Int, x: UnsafePointer<Double>, incX: Int) -> Double {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dsum(_backendIndex(n), x, _backendIndex(incX))
        #else
        _sum(n, x, incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func scsum(n: Int, x: UnsafePointer<Complex<Float>>, incX: Int) -> Float {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_scsum(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX))
        #else
        _sum(n, x, incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dzsum(n: Int, x: UnsafePointer<Complex<Double>>, incX: Int) -> Double {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dzsum(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _sum(n, x, incX)
        #endif
    }

    @inlinable
    @inline(always)
    public static func snrm2(n: Int, x: UnsafePointer<Float>, incX: Int) -> Float {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_snrm2(_backendIndex(n), x, _backendIndex(incX))
        #else
        _nrm2(n, x, incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dnrm2(n: Int, x: UnsafePointer<Double>, incX: Int) -> Double {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dnrm2(_backendIndex(n), x, _backendIndex(incX))
        #else
        _nrm2(n, x, incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func scnrm2(n: Int, x: UnsafePointer<Complex<Float>>, incX: Int) -> Float {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_scnrm2(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX))
        #else
        _nrm2(n, x, incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dznrm2(n: Int, x: UnsafePointer<Complex<Double>>, incX: Int) -> Double {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dznrm2(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _nrm2(n, x, incX)
        #endif
    }
}

// MARK: - Vector operations

extension BLAS {
    @inlinable
    @inline(always)
    public static func saxpy(
        n: Int, alpha: Float,
        x: UnsafePointer<Float>, incX: Int,
        y: UnsafeMutablePointer<Float>, incY: Int
    ) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_saxpy(_backendIndex(n), alpha, x, _backendIndex(incX), y, _backendIndex(incY))
        #else
        _axpy(n, alpha: alpha, x: x, incX: incX, y: y, incY: incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func daxpy(
        n: Int, alpha: Double,
        x: UnsafePointer<Double>, incX: Int,
        y: UnsafeMutablePointer<Double>, incY: Int
    ) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_daxpy(_backendIndex(n), alpha, x, _backendIndex(incX), y, _backendIndex(incY))
        #else
        _axpy(n, alpha: alpha, x: x, incX: incX, y: y, incY: incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func caxpy(
        n: Int, alpha: Complex<Float>,
        x: UnsafePointer<Complex<Float>>, incX: Int,
        y: UnsafeMutablePointer<Complex<Float>>, incY: Int
    ) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_caxpy(_backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY))
        }
        #else
        _axpy(n, alpha: alpha, x: x, incX: incX, y: y, incY: incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func zaxpy(
        n: Int, alpha: Complex<Double>,
        x: UnsafePointer<Complex<Double>>, incX: Int,
        y: UnsafeMutablePointer<Complex<Double>>, incY: Int
    ) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_zaxpy(_backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY))
        }
        #else
        _axpy(n, alpha: alpha, x: x, incX: incX, y: y, incY: incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func caxpyc(
        n: Int, alpha: Complex<Float>,
        x: UnsafePointer<Complex<Float>>, incX: Int,
        y: UnsafeMutablePointer<Complex<Float>>, incY: Int
    ) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        var alpha = alpha
        cblas_caxpyc(_backendIndex(n), &alpha, _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY))
        #else
        _axpy(n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, conjugateX: true)
        #endif
    }

    @inlinable
    @inline(always)
    public static func zaxpyc(
        n: Int, alpha: Complex<Double>,
        x: UnsafePointer<Complex<Double>>, incX: Int,
        y: UnsafeMutablePointer<Complex<Double>>, incY: Int
    ) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        var alpha = alpha
        cblas_zaxpyc(_backendIndex(n), &alpha, _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY))
        #else
        _axpy(n, alpha: alpha, x: x, incX: incX, y: y, incY: incY, conjugateX: true)
        #endif
    }

    @inlinable
    @inline(always)
    public static func scopy(n: Int, x: UnsafePointer<Float>, incX: Int, y: UnsafeMutablePointer<Float>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_scopy(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY))
        #else
        _copy(n, x: x, incX: incX, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dcopy(n: Int, x: UnsafePointer<Double>, incX: Int, y: UnsafeMutablePointer<Double>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dcopy(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY))
        #else
        _copy(n, x: x, incX: incX, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ccopy(n: Int, x: UnsafePointer<Complex<Float>>, incX: Int, y: UnsafeMutablePointer<Complex<Float>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_ccopy(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY))
        #else
        _copy(n, x: x, incX: incX, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zcopy(n: Int, x: UnsafePointer<Complex<Double>>, incX: Int, y: UnsafeMutablePointer<Complex<Double>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_zcopy(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY))
        #else
        _copy(n, x: x, incX: incX, y: y, incY: incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func sswap(n: Int, x: UnsafeMutablePointer<Float>, incX: Int, y: UnsafeMutablePointer<Float>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sswap(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY))
        #else
        _swap(n, x: x, incX: incX, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dswap(n: Int, x: UnsafeMutablePointer<Double>, incX: Int, y: UnsafeMutablePointer<Double>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dswap(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY))
        #else
        _swap(n, x: x, incX: incX, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cswap(n: Int, x: UnsafeMutablePointer<Complex<Float>>, incX: Int, y: UnsafeMutablePointer<Complex<Float>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_cswap(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY))
        #else
        _swap(n, x: x, incX: incX, y: y, incY: incY)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zswap(n: Int, x: UnsafeMutablePointer<Complex<Double>>, incX: Int, y: UnsafeMutablePointer<Complex<Double>>, incY: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_zswap(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY))
        #else
        _swap(n, x: x, incX: incX, y: y, incY: incY)
        #endif
    }

    @inlinable
    @inline(always)
    public static func srot(n: Int, x: UnsafeMutablePointer<Float>, incX: Int, y: UnsafeMutablePointer<Float>, incY: Int, c: Float, s: Float) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_srot(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY), c, s)
        #else
        _rot(n, x: x, incX: incX, y: y, incY: incY, c: c, s: s)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func drot(n: Int, x: UnsafeMutablePointer<Double>, incX: Int, y: UnsafeMutablePointer<Double>, incY: Int, c: Double, s: Double) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_drot(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY), c, s)
        #else
        _rot(n, x: x, incX: incX, y: y, incY: incY, c: c, s: s)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func csrot(n: Int, x: UnsafeMutablePointer<Complex<Float>>, incX: Int, y: UnsafeMutablePointer<Complex<Float>>, incY: Int, c: Float, s: Float) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_csrot(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX), _complexFloatPointer(y), _backendIndex(incY), c, s)
        #else
        _rot(n, x: x, incX: incX, y: y, incY: incY, c: c, s: s)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zdrot(n: Int, x: UnsafeMutablePointer<Complex<Double>>, incX: Int, y: UnsafeMutablePointer<Complex<Double>>, incY: Int, c: Double, s: Double) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_zdrot(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX), _complexDoublePointer(y), _backendIndex(incY), c, s)
        #else
        _rot(n, x: x, incX: incX, y: y, incY: incY, c: c, s: s)
        #endif
    }

    @inlinable
    @inline(always)
    public static func srotg(a: UnsafeMutablePointer<Float>, b: UnsafeMutablePointer<Float>, c: UnsafeMutablePointer<Float>, s: UnsafeMutablePointer<Float>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_srotg(a, b, c, s)
        #else
        _rotg(a: a, b: b, c: c, s: s)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func drotg(a: UnsafeMutablePointer<Double>, b: UnsafeMutablePointer<Double>, c: UnsafeMutablePointer<Double>, s: UnsafeMutablePointer<Double>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_drotg(a, b, c, s)
        #else
        _rotg(a: a, b: b, c: c, s: s)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func crotg(a: UnsafeMutablePointer<Complex<Float>>, b: UnsafeMutablePointer<Complex<Float>>, c: UnsafeMutablePointer<Float>, s: UnsafeMutablePointer<Complex<Float>>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_crotg(_complexFloatPointer(a), _complexFloatPointer(b), c, _complexFloatPointer(s))
        #else
        _complexRotg(a: a, b: UnsafePointer(b), c: c, s: s)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zrotg(a: UnsafeMutablePointer<Complex<Double>>, b: UnsafeMutablePointer<Complex<Double>>, c: UnsafeMutablePointer<Double>, s: UnsafeMutablePointer<Complex<Double>>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_zrotg(_complexDoublePointer(a), _complexDoublePointer(b), c, _complexDoublePointer(s))
        #else
        _complexRotg(a: a, b: UnsafePointer(b), c: c, s: s)
        #endif
    }

    @inlinable
    @inline(always)
    public static func srotm(n: Int, x: UnsafeMutablePointer<Float>, incX: Int, y: UnsafeMutablePointer<Float>, incY: Int, parameters: UnsafePointer<Float>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_srotm(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY), parameters)
        #else
        _rotm(n, x: x, incX: incX, y: y, incY: incY, parameters: parameters)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func drotm(n: Int, x: UnsafeMutablePointer<Double>, incX: Int, y: UnsafeMutablePointer<Double>, incY: Int, parameters: UnsafePointer<Double>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_drotm(_backendIndex(n), x, _backendIndex(incX), y, _backendIndex(incY), parameters)
        #else
        _rotm(n, x: x, incX: incX, y: y, incY: incY, parameters: parameters)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func srotmg(d1: UnsafeMutablePointer<Float>, d2: UnsafeMutablePointer<Float>, b1: UnsafeMutablePointer<Float>, b2: Float, parameters: UnsafeMutablePointer<Float>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_srotmg(d1, d2, b1, b2, parameters)
        #else
        _rotmg(d1: d1, d2: d2, b1: b1, b2: b2, parameters: parameters)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func drotmg(d1: UnsafeMutablePointer<Double>, d2: UnsafeMutablePointer<Double>, b1: UnsafeMutablePointer<Double>, b2: Double, parameters: UnsafeMutablePointer<Double>) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_drotmg(d1, d2, b1, b2, parameters)
        #else
        _rotmg(d1: d1, d2: d2, b1: b1, b2: b2, parameters: parameters)
        #endif
    }

    @inlinable
    @inline(always)
    public static func sscal(n: Int, alpha: Float, x: UnsafeMutablePointer<Float>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sscal(_backendIndex(n), alpha, x, _backendIndex(incX))
        #else
        _scal(n, alpha: alpha, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dscal(n: Int, alpha: Double, x: UnsafeMutablePointer<Double>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dscal(_backendIndex(n), alpha, x, _backendIndex(incX))
        #else
        _scal(n, alpha: alpha, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func cscal(n: Int, alpha: Complex<Float>, x: UnsafeMutablePointer<Complex<Float>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_cscal(_backendIndex(n), _complexFloatPointer(alpha), _complexFloatPointer(x), _backendIndex(incX))
        }
        #else
        _scal(n, alpha: alpha, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zscal(n: Int, alpha: Complex<Double>, x: UnsafeMutablePointer<Complex<Double>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        withUnsafePointer(to: alpha) { alpha in
            cblas_zscal(_backendIndex(n), _complexDoublePointer(alpha), _complexDoublePointer(x), _backendIndex(incX))
        }
        #else
        _scal(n, alpha: alpha, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func csscal(n: Int, alpha: Float, x: UnsafeMutablePointer<Complex<Float>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_csscal(_backendIndex(n), alpha, _complexFloatPointer(x), _backendIndex(incX))
        #else
        _realScal(n, alpha: alpha, x: x, incX: incX)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func zdscal(n: Int, alpha: Double, x: UnsafeMutablePointer<Complex<Double>>, incX: Int) {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_zdscal(_backendIndex(n), alpha, _complexDoublePointer(x), _backendIndex(incX))
        #else
        _realScal(n, alpha: alpha, x: x, incX: incX)
        #endif
    }
}

// MARK: - Reductions

extension BLAS {
    @inlinable
    @inline(always)
    public static func isamax(n: Int, x: UnsafePointer<Float>, incX: Int) -> Int {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_isamax(_backendIndex(n), x, _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: true, minimum: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func idamax(n: Int, x: UnsafePointer<Double>, incX: Int) -> Int {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_idamax(_backendIndex(n), x, _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: true, minimum: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func icamax(n: Int, x: UnsafePointer<Complex<Float>>, incX: Int) -> Int {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_icamax(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: true, minimum: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func izamax(n: Int, x: UnsafePointer<Complex<Double>>, incX: Int) -> Int {
        #if (canImport(COpenBLAS) || (canImport(Accelerate) && os(macOS))) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_izamax(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: true, minimum: false)
        #endif
    }

    @inlinable
    @inline(always)
    public static func isamin(n: Int, x: UnsafePointer<Float>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_isamin(_backendIndex(n), x, _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: true, minimum: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func idamin(n: Int, x: UnsafePointer<Double>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_idamin(_backendIndex(n), x, _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: true, minimum: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func icamin(n: Int, x: UnsafePointer<Complex<Float>>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_icamin(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: true, minimum: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func izamin(n: Int, x: UnsafePointer<Complex<Double>>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_izamin(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: true, minimum: true)
        #endif
    }

    @inlinable
    @inline(always)
    public static func samax(n: Int, x: UnsafePointer<Float>, incX: Int) -> Float {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_samax(_backendIndex(n), x, _backendIndex(incX))
        #else
        _extremeValue(n, x, incX, absolute: true, minimum: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func damax(n: Int, x: UnsafePointer<Double>, incX: Int) -> Double {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_damax(_backendIndex(n), x, _backendIndex(incX))
        #else
        _extremeValue(n, x, incX, absolute: true, minimum: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dzamax(n: Int, x: UnsafePointer<Complex<Double>>, incX: Int) -> Double {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dzamax(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _extremeValue(n, x, incX, absolute: true, minimum: false)
        #endif
    }

    @inlinable
    @inline(always)
    public static func samin(n: Int, x: UnsafePointer<Float>, incX: Int) -> Float {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_samin(_backendIndex(n), x, _backendIndex(incX))
        #else
        _extremeValue(n, x, incX, absolute: true, minimum: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func damin(n: Int, x: UnsafePointer<Double>, incX: Int) -> Double {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_damin(_backendIndex(n), x, _backendIndex(incX))
        #else
        _extremeValue(n, x, incX, absolute: true, minimum: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func scamin(n: Int, x: UnsafePointer<Complex<Float>>, incX: Int) -> Float {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_scamin(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX))
        #else
        _extremeValue(n, x, incX, absolute: true, minimum: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func dzamin(n: Int, x: UnsafePointer<Complex<Double>>, incX: Int) -> Double {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dzamin(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX))
        #else
        _extremeValue(n, x, incX, absolute: true, minimum: true)
        #endif
    }

    @inlinable
    @inline(always)
    public static func ismax(n: Int, x: UnsafePointer<Float>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_ismax(_backendIndex(n), x, _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: false, minimum: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func idmax(n: Int, x: UnsafePointer<Double>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_idmax(_backendIndex(n), x, _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: false, minimum: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func icmax(n: Int, x: UnsafePointer<Complex<Float>>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_icmax(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: false, minimum: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func izmax(n: Int, x: UnsafePointer<Complex<Double>>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_izmax(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: false, minimum: false)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func ismin(n: Int, x: UnsafePointer<Float>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_ismin(_backendIndex(n), x, _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: false, minimum: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func idmin(n: Int, x: UnsafePointer<Double>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_idmin(_backendIndex(n), x, _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: false, minimum: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func icmin(n: Int, x: UnsafePointer<Complex<Float>>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_icmin(_backendIndex(n), _complexFloatPointer(x), _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: false, minimum: true)
        #endif
    }
    
    @inlinable
    @inline(always)
    public static func izmin(n: Int, x: UnsafePointer<Complex<Double>>, incX: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(cblas_izmin(_backendIndex(n), _complexDoublePointer(x), _backendIndex(incX)))
        #else
        _extremeIndex(n, x, incX, absolute: false, minimum: true)
        #endif
    }
}
