// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

#if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
import COpenBLAS
#endif

import ComplexModule
import RealModule

extension BLAS {
    @inlinable
    public static func sgemmBatch(
        layout: Layout,
        transposeA: UnsafePointer<Transpose>, transposeB: UnsafePointer<Transpose>,
        m: UnsafePointer<Int>, n: UnsafePointer<Int>, k: UnsafePointer<Int>,
        alpha: UnsafePointer<Float>,
        a: UnsafePointer<UnsafePointer<Float>>, lda: UnsafePointer<Int>,
        b: UnsafePointer<UnsafePointer<Float>>, ldb: UnsafePointer<Int>,
        beta: UnsafePointer<Float>,
        c: UnsafePointer<UnsafeMutablePointer<Float>>, ldc: UnsafePointer<Int>,
        groupCount: Int, groupSize: UnsafePointer<Int>
    ) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        _ = _backendIndex(groupCount)
        var offset = 0
        for group in 0..<_blasCount(groupCount) {
            precondition(groupSize[group] >= 0, "BLAS batch group sizes must be nonnegative")
            let count = _blasCount(groupSize[group])
            var ta = transposeA[group]._cblas
            var tb = transposeB[group]._cblas
            var mm = _backendIndex(m[group])
            var nn = _backendIndex(n[group])
            var kk = _backendIndex(k[group])
            var leadingA = _backendIndex(lda[group])
            var leadingB = _backendIndex(ldb[group])
            var leadingC = _backendIndex(ldc[group])
            var size = _backendIndex(groupSize[group])
            if count == 0 { continue }
            var aPointers = Array<UnsafePointer<Float>?>(repeating: nil, count: count)
            var bPointers = Array<UnsafePointer<Float>?>(repeating: nil, count: count)
            var cPointers = Array<UnsafeMutablePointer<Float>?>(repeating: nil, count: count)
            for item in 0..<count {
                aPointers[item] = a[offset + item]
                bPointers[item] = b[offset + item]
                cPointers[item] = c[offset + item]
            }
            aPointers.withUnsafeMutableBufferPointer { aArray in
                bPointers.withUnsafeMutableBufferPointer { bArray in
                    cPointers.withUnsafeMutableBufferPointer { cArray in
                        cblas_sgemm_batch(layout._cblas, &ta, &tb, &mm, &nn, &kk, alpha.advanced(by: group), aArray.baseAddress, &leadingA, bArray.baseAddress, &leadingB, beta.advanced(by: group), cArray.baseAddress, &leadingC, 1, &size)
                    }
                }
            }
            offset += count
        }
        #else
        var offset = 0
        for group in 0..<_blasCount(groupCount) {
            for _ in 0..<_blasCount(groupSize[group]) {
                sgemm(layout: layout, transposeA: transposeA[group], transposeB: transposeB[group], m: m[group], n: n[group], k: k[group], alpha: alpha[group], a: a[offset], lda: lda[group], b: b[offset], ldb: ldb[group], beta: beta[group], c: c[offset], ldc: ldc[group])
                offset += 1
            }
        }
        #endif
    }

    @inlinable
    public static func dgemmBatch(
        layout: Layout,
        transposeA: UnsafePointer<Transpose>, transposeB: UnsafePointer<Transpose>,
        m: UnsafePointer<Int>, n: UnsafePointer<Int>, k: UnsafePointer<Int>,
        alpha: UnsafePointer<Double>,
        a: UnsafePointer<UnsafePointer<Double>>, lda: UnsafePointer<Int>,
        b: UnsafePointer<UnsafePointer<Double>>, ldb: UnsafePointer<Int>,
        beta: UnsafePointer<Double>,
        c: UnsafePointer<UnsafeMutablePointer<Double>>, ldc: UnsafePointer<Int>,
        groupCount: Int, groupSize: UnsafePointer<Int>
    ) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        _ = _backendIndex(groupCount)
        var offset = 0
        for group in 0..<_blasCount(groupCount) {
            precondition(groupSize[group] >= 0, "BLAS batch group sizes must be nonnegative")
            let count = _blasCount(groupSize[group])
            var ta = transposeA[group]._cblas
            var tb = transposeB[group]._cblas
            var mm = _backendIndex(m[group])
            var nn = _backendIndex(n[group])
            var kk = _backendIndex(k[group])
            var leadingA = _backendIndex(lda[group])
            var leadingB = _backendIndex(ldb[group])
            var leadingC = _backendIndex(ldc[group])
            var size = _backendIndex(groupSize[group])
            if count == 0 { continue }
            var aPointers = Array<UnsafePointer<Double>?>(repeating: nil, count: count)
            var bPointers = Array<UnsafePointer<Double>?>(repeating: nil, count: count)
            var cPointers = Array<UnsafeMutablePointer<Double>?>(repeating: nil, count: count)
            for item in 0..<count {
                aPointers[item] = a[offset + item]
                bPointers[item] = b[offset + item]
                cPointers[item] = c[offset + item]
            }
            aPointers.withUnsafeMutableBufferPointer { aArray in
                bPointers.withUnsafeMutableBufferPointer { bArray in
                    cPointers.withUnsafeMutableBufferPointer { cArray in
                        cblas_dgemm_batch(layout._cblas, &ta, &tb, &mm, &nn, &kk, alpha.advanced(by: group), aArray.baseAddress, &leadingA, bArray.baseAddress, &leadingB, beta.advanced(by: group), cArray.baseAddress, &leadingC, 1, &size)
                    }
                }
            }
            offset += count
        }
        #else
        var offset = 0
        for group in 0..<_blasCount(groupCount) {
            for _ in 0..<_blasCount(groupSize[group]) {
                dgemm(layout: layout, transposeA: transposeA[group], transposeB: transposeB[group], m: m[group], n: n[group], k: k[group], alpha: alpha[group], a: a[offset], lda: lda[group], b: b[offset], ldb: ldb[group], beta: beta[group], c: c[offset], ldc: ldc[group])
                offset += 1
            }
        }
        #endif
    }

    @inlinable
    public static func cgemmBatch(
        layout: Layout,
        transposeA: UnsafePointer<Transpose>, transposeB: UnsafePointer<Transpose>,
        m: UnsafePointer<Int>, n: UnsafePointer<Int>, k: UnsafePointer<Int>,
        alpha: UnsafePointer<Complex<Float>>,
        a: UnsafePointer<UnsafePointer<Complex<Float>>>, lda: UnsafePointer<Int>,
        b: UnsafePointer<UnsafePointer<Complex<Float>>>, ldb: UnsafePointer<Int>,
        beta: UnsafePointer<Complex<Float>>,
        c: UnsafePointer<UnsafeMutablePointer<Complex<Float>>>, ldc: UnsafePointer<Int>,
        groupCount: Int, groupSize: UnsafePointer<Int>
    ) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        _ = _backendIndex(groupCount)
        var offset = 0
        for group in 0..<_blasCount(groupCount) {
            precondition(groupSize[group] >= 0, "BLAS batch group sizes must be nonnegative")
            let count = _blasCount(groupSize[group])
            var ta = transposeA[group]._cblas
            var tb = transposeB[group]._cblas
            var mm = _backendIndex(m[group])
            var nn = _backendIndex(n[group])
            var kk = _backendIndex(k[group])
            var leadingA = _backendIndex(lda[group])
            var leadingB = _backendIndex(ldb[group])
            var leadingC = _backendIndex(ldc[group])
            var size = _backendIndex(groupSize[group])
            if count == 0 { continue }
            var aPointers = Array<UnsafeRawPointer?>(repeating: nil, count: count)
            var bPointers = Array<UnsafeRawPointer?>(repeating: nil, count: count)
            var cPointers = Array<UnsafeMutableRawPointer?>(repeating: nil, count: count)
            for item in 0..<count {
                aPointers[item] = UnsafeRawPointer(a[offset + item])
                bPointers[item] = UnsafeRawPointer(b[offset + item])
                cPointers[item] = UnsafeMutableRawPointer(c[offset + item])
            }
            aPointers.withUnsafeMutableBufferPointer { aArray in
                bPointers.withUnsafeMutableBufferPointer { bArray in
                    cPointers.withUnsafeMutableBufferPointer { cArray in
                        cblas_cgemm_batch(layout._cblas, &ta, &tb, &mm, &nn, &kk, UnsafeRawPointer(alpha.advanced(by: group)), aArray.baseAddress, &leadingA, bArray.baseAddress, &leadingB, UnsafeRawPointer(beta.advanced(by: group)), cArray.baseAddress, &leadingC, 1, &size)
                    }
                }
            }
            offset += count
        }
        #else
        var offset = 0
        for group in 0..<_blasCount(groupCount) {
            for _ in 0..<_blasCount(groupSize[group]) {
                cgemm(layout: layout, transposeA: transposeA[group], transposeB: transposeB[group], m: m[group], n: n[group], k: k[group], alpha: alpha[group], a: a[offset], lda: lda[group], b: b[offset], ldb: ldb[group], beta: beta[group], c: c[offset], ldc: ldc[group])
                offset += 1
            }
        }
        #endif
    }

    @inlinable
    public static func zgemmBatch(
        layout: Layout,
        transposeA: UnsafePointer<Transpose>, transposeB: UnsafePointer<Transpose>,
        m: UnsafePointer<Int>, n: UnsafePointer<Int>, k: UnsafePointer<Int>,
        alpha: UnsafePointer<Complex<Double>>,
        a: UnsafePointer<UnsafePointer<Complex<Double>>>, lda: UnsafePointer<Int>,
        b: UnsafePointer<UnsafePointer<Complex<Double>>>, ldb: UnsafePointer<Int>,
        beta: UnsafePointer<Complex<Double>>,
        c: UnsafePointer<UnsafeMutablePointer<Complex<Double>>>, ldc: UnsafePointer<Int>,
        groupCount: Int, groupSize: UnsafePointer<Int>
    ) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        _ = _backendIndex(groupCount)
        var offset = 0
        for group in 0..<_blasCount(groupCount) {
            precondition(groupSize[group] >= 0, "BLAS batch group sizes must be nonnegative")
            let count = _blasCount(groupSize[group])
            var ta = transposeA[group]._cblas
            var tb = transposeB[group]._cblas
            var mm = _backendIndex(m[group])
            var nn = _backendIndex(n[group])
            var kk = _backendIndex(k[group])
            var leadingA = _backendIndex(lda[group])
            var leadingB = _backendIndex(ldb[group])
            var leadingC = _backendIndex(ldc[group])
            var size = _backendIndex(groupSize[group])
            if count == 0 { continue }
            var aPointers = Array<UnsafeRawPointer?>(repeating: nil, count: count)
            var bPointers = Array<UnsafeRawPointer?>(repeating: nil, count: count)
            var cPointers = Array<UnsafeMutableRawPointer?>(repeating: nil, count: count)
            for item in 0..<count {
                aPointers[item] = UnsafeRawPointer(a[offset + item])
                bPointers[item] = UnsafeRawPointer(b[offset + item])
                cPointers[item] = UnsafeMutableRawPointer(c[offset + item])
            }
            aPointers.withUnsafeMutableBufferPointer { aArray in
                bPointers.withUnsafeMutableBufferPointer { bArray in
                    cPointers.withUnsafeMutableBufferPointer { cArray in
                        cblas_zgemm_batch(layout._cblas, &ta, &tb, &mm, &nn, &kk, UnsafeRawPointer(alpha.advanced(by: group)), aArray.baseAddress, &leadingA, bArray.baseAddress, &leadingB, UnsafeRawPointer(beta.advanced(by: group)), cArray.baseAddress, &leadingC, 1, &size)
                    }
                }
            }
            offset += count
        }
        #else
        var offset = 0
        for group in 0..<_blasCount(groupCount) {
            for _ in 0..<_blasCount(groupSize[group]) {
                zgemm(layout: layout, transposeA: transposeA[group], transposeB: transposeB[group], m: m[group], n: n[group], k: k[group], alpha: alpha[group], a: a[offset], lda: lda[group], b: b[offset], ldb: ldb[group], beta: beta[group], c: c[offset], ldc: ldc[group])
                offset += 1
            }
        }
        #endif
    }
}

extension BLAS {
    @inlinable
    public static func sgemmBatchStrided(layout: Layout, transposeA: Transpose, transposeB: Transpose, m: Int, n: Int, k: Int, alpha: Float, a: UnsafePointer<Float>, lda: Int, strideA: Int, b: UnsafePointer<Float>, ldb: Int, strideB: Int, beta: Float, c: UnsafeMutablePointer<Float>, ldc: Int, strideC: Int, groupSize: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_sgemm_batch_strided(layout._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(k), alpha, a, _backendIndex(lda), _backendIndex(strideA), b, _backendIndex(ldb), _backendIndex(strideB), beta, c, _backendIndex(ldc), _backendIndex(strideC), _backendIndex(groupSize))
        #else
        for batch in 0..<_blasCount(groupSize) {
            sgemm(layout: layout, transposeA: transposeA, transposeB: transposeB, m: m, n: n, k: k, alpha: alpha, a: a.advanced(by: batch * strideA), lda: lda, b: b.advanced(by: batch * strideB), ldb: ldb, beta: beta, c: c.advanced(by: batch * strideC), ldc: ldc)
        }
        #endif
    }
    
    @inlinable
    public static func dgemmBatchStrided(layout: Layout, transposeA: Transpose, transposeB: Transpose, m: Int, n: Int, k: Int, alpha: Double, a: UnsafePointer<Double>, lda: Int, strideA: Int, b: UnsafePointer<Double>, ldb: Int, strideB: Int, beta: Double, c: UnsafeMutablePointer<Double>, ldc: Int, strideC: Int, groupSize: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        cblas_dgemm_batch_strided(layout._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(k), alpha, a, _backendIndex(lda), _backendIndex(strideA), b, _backendIndex(ldb), _backendIndex(strideB), beta, c, _backendIndex(ldc), _backendIndex(strideC), _backendIndex(groupSize))
        #else
        for batch in 0..<_blasCount(groupSize) {
            dgemm(layout: layout, transposeA: transposeA, transposeB: transposeB, m: m, n: n, k: k, alpha: alpha, a: a.advanced(by: batch * strideA), lda: lda, b: b.advanced(by: batch * strideB), ldb: ldb, beta: beta, c: c.advanced(by: batch * strideC), ldc: ldc)
        }
        #endif
    }
    
    @inlinable
    public static func cgemmBatchStrided(layout: Layout, transposeA: Transpose, transposeB: Transpose, m: Int, n: Int, k: Int, alpha: Complex<Float>, a: UnsafePointer<Complex<Float>>, lda: Int, strideA: Int, b: UnsafePointer<Complex<Float>>, ldb: Int, strideB: Int, beta: Complex<Float>, c: UnsafeMutablePointer<Complex<Float>>, ldc: Int, strideC: Int, groupSize: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        var alpha = alpha; var beta = beta
        cblas_cgemm_batch_strided(layout._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(k), &alpha, _complexFloatPointer(a), _backendIndex(lda), _backendIndex(strideA), _complexFloatPointer(b), _backendIndex(ldb), _backendIndex(strideB), &beta, _complexFloatPointer(c), _backendIndex(ldc), _backendIndex(strideC), _backendIndex(groupSize))
        #else
        for batch in 0..<_blasCount(groupSize) {
            cgemm(layout: layout, transposeA: transposeA, transposeB: transposeB, m: m, n: n, k: k, alpha: alpha, a: a.advanced(by: batch * strideA), lda: lda, b: b.advanced(by: batch * strideB), ldb: ldb, beta: beta, c: c.advanced(by: batch * strideC), ldc: ldc)
        }
        #endif
    }
    
    @inlinable
    public static func zgemmBatchStrided(layout: Layout, transposeA: Transpose, transposeB: Transpose, m: Int, n: Int, k: Int, alpha: Complex<Double>, a: UnsafePointer<Complex<Double>>, lda: Int, strideA: Int, b: UnsafePointer<Complex<Double>>, ldb: Int, strideB: Int, beta: Complex<Double>, c: UnsafeMutablePointer<Complex<Double>>, ldc: Int, strideC: Int, groupSize: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        var alpha = alpha; var beta = beta
        cblas_zgemm_batch_strided(layout._cblas, transposeA._cblas, transposeB._cblas, _backendIndex(m), _backendIndex(n), _backendIndex(k), &alpha, _complexDoublePointer(a), _backendIndex(lda), _backendIndex(strideA), _complexDoublePointer(b), _backendIndex(ldb), _backendIndex(strideB), &beta, _complexDoublePointer(c), _backendIndex(ldc), _backendIndex(strideC), _backendIndex(groupSize))
        #else
        for batch in 0..<_blasCount(groupSize) {
            zgemm(layout: layout, transposeA: transposeA, transposeB: transposeB, m: m, n: n, k: k, alpha: alpha, a: a.advanced(by: batch * strideA), lda: lda, b: b.advanced(by: batch * strideB), ldb: ldb, beta: beta, c: c.advanced(by: batch * strideC), ldc: ldc)
        }
        #endif
    }
}
