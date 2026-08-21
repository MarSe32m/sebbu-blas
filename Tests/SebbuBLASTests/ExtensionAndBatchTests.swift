import ComplexModule
import RealModule
import SebbuBLAS
import Testing

@Suite("OpenBLAS extensions")
struct ExtensionAndBatchTests {
    @Test
    func axpbyUsesBothScalarsAndLogicalStrides() {
        let x = [1.0, 99, 2, 99, 3]
        var y = [10.0, 88, 20, 88, 30]
        BLAS.daxpby(
            n: 3, alpha: 2,
            x: x, incX: 2,
            beta: -1, y: &y, incY: 2
        )
        expectApproximatelyEqual(y, [-8, 88, -16, 88, -24])

        let complexX = [Complex<Double>(1, 1), Complex(2, -1)]
        var complexY = [Complex<Double>(3, 0), Complex(-1, 2)]
        BLAS.zaxpby(
            n: 2, alpha: Complex(0, 1),
            x: complexX, incX: 1,
            beta: Complex(2, 0), y: &complexY, incY: 1
        )
        expectApproximatelyEqual(
            complexY,
            [Complex(5, 1), Complex(-1, 6)]
        )
    }

    @Test
    func complexMatrixCopyConjugatesAndTransposes() {
        let source: [Complex<Double>] = [
            Complex(1, 1), Complex(2, -1), Complex(3, 2),
            Complex(4, 0), Complex(5, -2), Complex(6, 1),
        ]
        var destination = [Complex<Double>](repeating: .zero, count: 6)
        BLAS.zomatcopy(
            layout: .rowMajor, transpose: .conjugateTranspose,
            rows: 2, columns: 3, alpha: .one,
            a: source, lda: 3,
            b: &destination, ldb: 2
        )
        expectApproximatelyEqual(
            destination,
            [
                Complex(1, -1), Complex(4, 0),
                Complex(2, 1), Complex(5, 2),
                Complex(3, -2), Complex(6, -1),
            ]
        )
    }

    @Test
    func geaddSupportsTransposedInputAndBeta() {
        let a = [1.0, 2, 3, 4, 5, 6]
        var c = [Double](repeating: 1, count: 6)
        BLAS.dgeadd(
            layout: .rowMajor,
            transposeA: .transpose, transposeC: .noTranspose,
            rows: 3, columns: 2,
            alpha: 2, a: a, lda: 3,
            beta: 3, c: &c, ldc: 2
        )
        expectApproximatelyEqual(c, [5, 11, 7, 13, 9, 15])
    }

    @Test(
        .enabled(
            if: !BLAS.getConfig().contains("OpenBLAS"),
            "OpenBLAS 0.3.34's batch extension crashes before returning; the Swift and Accelerate paths remain covered."
        )
    )
    func groupedGemmUsesPerGroupScalarsAndOffsets() {
        let a = [2.0, 3, 4]
        let b = [5.0, 6, 7]
        var c = [1.0, 1, 1]
        a.withUnsafeBufferPointer { a in
            b.withUnsafeBufferPointer { b in
                c.withUnsafeMutableBufferPointer { c in
                    let aPointers = [
                        a.baseAddress!, a.baseAddress!.advanced(by: 1),
                        a.baseAddress!.advanced(by: 2),
                    ]
                    let bPointers = [
                        b.baseAddress!, b.baseAddress!.advanced(by: 1),
                        b.baseAddress!.advanced(by: 2),
                    ]
                    let cPointers = [
                        c.baseAddress!, c.baseAddress!.advanced(by: 1),
                        c.baseAddress!.advanced(by: 2),
                    ]
                    BLAS.dgemmBatch(
                        layout: .rowMajor,
                        transposeA: [.noTranspose, .noTranspose],
                        transposeB: [.noTranspose, .noTranspose],
                        m: [1, 1], n: [1, 1], k: [1, 1],
                        alpha: [1, 2],
                        a: aPointers, lda: [1, 1],
                        b: bPointers, ldb: [1, 1],
                        beta: [0, 1],
                        c: cPointers, ldc: [1, 1],
                        groupCount: 2, groupSize: [2, 1]
                    )
                }
            }
        }
        expectApproximatelyEqual(c, [10, 18, 57])
    }

    @Test(
        .enabled(
            if: !BLAS.getConfig().contains("OpenBLAS"),
            "OpenBLAS 0.3.34's batch extension crashes before returning; the Swift and Accelerate paths remain covered."
        )
    )
    func stridedGemmLeavesPaddingUntouched() {
        let a = [
            1.0, 0, 0, 1, -1,
            2, 0, 0, 3, -1,
        ]
        let b = [
            1.0, 2, 3, 4, -1,
            1, 2, 3, 4, -1,
        ]
        var c = [
            0.0, 0, 0, 0, 99,
            0, 0, 0, 0, 99,
        ]
        BLAS.dgemmBatchStrided(
            layout: .rowMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 2, n: 2, k: 2,
            alpha: 1, a: a, lda: 2, strideA: 5,
            b: b, ldb: 2, strideB: 5,
            beta: 0, c: &c, ldc: 2, strideC: 5,
            groupSize: 2
        )
        expectApproximatelyEqual(
            c,
            [1, 2, 3, 4, 99, 2, 4, 9, 12, 99]
        )
    }
}

@Suite("BLAS runtime metadata")
struct RuntimeTests {
    @Test
    func backendMetadataIsUsableOnEveryBackend() {
        #expect(BLAS.getNumThreads() >= 1)
        #expect(BLAS.getNumProcs() >= 1)
        #expect(!BLAS.getConfig().isEmpty)
        #expect(!BLAS.getCorename().isEmpty)
        #expect(
            BLAS.getParallel() == BLAS.Parallelization.sequential.rawValue
                || BLAS.getParallel() == BLAS.Parallelization.thread.rawValue
                || BLAS.getParallel() == BLAS.Parallelization.openMP.rawValue
        )
    }
}
