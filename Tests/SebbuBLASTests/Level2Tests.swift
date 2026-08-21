import ComplexModule
import RealModule
import SebbuBLAS
import Testing

@Suite("Level 2 BLAS")
struct Level2Tests {
    @Test
    func generalMatrixVectorProductsCoverLayoutsAndConjugation() {
        let rowMajor = [1.0, 2, 3, 4, 5, 6]
        let columnMajor = [1.0, 4, 2, 5, 3, 6]
        let x = [1.0, -1, 2]
        var rowResult = [10.0, 20]
        var columnResult = rowResult

        BLAS.dgemv(
            layout: .rowMajor, transpose: .noTranspose,
            m: 2, n: 3, alpha: 2,
            a: rowMajor, lda: 3,
            x: x, incX: 1,
            beta: 0.5, y: &rowResult, incY: 1
        )
        BLAS.dgemv(
            layout: .columnMajor, transpose: .noTranspose,
            m: 2, n: 3, alpha: 2,
            a: columnMajor, lda: 2,
            x: x, incX: 1,
            beta: 0.5, y: &columnResult, incY: 1
        )
        expectApproximatelyEqual(rowResult, [15, 32])
        expectApproximatelyEqual(columnResult, rowResult)

        let complexMatrix: [Complex<Double>] = [
            Complex(1, 1), Complex(2, 0),
            Complex(3, -1), Complex(4, 2),
        ]
        let complexX: [Complex<Double>] = [Complex(1, 0), .zero]
        var complexResult = [Complex<Double>](repeating: .zero, count: 2)
        BLAS.zgemv(
            layout: .rowMajor, transpose: .conjugateTranspose,
            m: 2, n: 2, alpha: .one,
            a: complexMatrix, lda: 2,
            x: complexX, incX: 1,
            beta: .zero, y: &complexResult, incY: 1
        )
        expectApproximatelyEqual(
            complexResult,
            [Complex(1, -1), Complex(2, 0)]
        )
    }

    @Test
    func generalBandMatrixVectorProductUsesBandStorage() {
        // Row-major storage for
        // [[1, 2, 0, 0], [3, 4, 5, 0], [0, 6, 7, 8]].
        let band = [0.0, 1, 2, 3, 4, 5, 6, 7, 8]
        let x = [1.0, 1, 1, 1]
        var y = [Double](repeating: 0, count: 3)
        BLAS.dgbmv(
            layout: .rowMajor, transpose: .noTranspose,
            m: 3, n: 4, kl: 1, ku: 1,
            alpha: 1, a: band, lda: 3,
            x: x, incX: 1, beta: 0, y: &y, incY: 1
        )
        expectApproximatelyEqual(y, [3, 12, 21])

        var transposed = [Double](repeating: 0, count: 4)
        BLAS.dgbmv(
            layout: .rowMajor, transpose: .transpose,
            m: 3, n: 4, kl: 1, ku: 1,
            alpha: 1, a: band, lda: 3,
            x: [1.0, 1, 1], incX: 1,
            beta: 0, y: &transposed, incY: 1
        )
        expectApproximatelyEqual(transposed, [4, 12, 12, 8])
    }

    @Test
    func symmetricAndHermitianBandProductsReflectOnlyStoredElements() {
        let symmetricBand = [1.0, 2, 3, 4, 5, 0]
        var symmetricResult = [Double](repeating: 0, count: 3)
        BLAS.dsbmv(
            layout: .rowMajor, triangle: .upper,
            n: 3, k: 1, alpha: 1,
            a: symmetricBand, lda: 2,
            x: [1.0, 1, 1], incX: 1,
            beta: 0, y: &symmetricResult, incY: 1
        )
        expectApproximatelyEqual(symmetricResult, [3, 9, 9])

        let hermitianBand: [Complex<Double>] = [
            Complex(1, 9), Complex(2, 1),
            Complex(3, 8), Complex(4, -2),
            Complex(5, 7), .zero,
        ]
        var hermitianResult = [Complex<Double>](repeating: .zero, count: 3)
        BLAS.zhbmv(
            layout: .rowMajor, triangle: .upper,
            n: 3, k: 1, alpha: .one,
            a: hermitianBand, lda: 2,
            x: [Complex<Double>.one, .one, .one], incX: 1,
            beta: .zero, y: &hermitianResult, incY: 1
        )
        expectApproximatelyEqual(
            hermitianResult,
            [Complex(3, 1), Complex(9, -3), Complex(9, 2)]
        )
    }

    @Test
    func triangularDenseMultiplyAndSolveAreInverseOperations() {
        // Unit-lower [[1, 0], [2, 1]]; stored diagonal values must be ignored.
        let matrix = [99.0, 0, 2, -17]
        let original = [3.0, 4]
        var x = original
        BLAS.dtrmv(
            layout: .rowMajor, triangle: .lower,
            transpose: .noTranspose, diagonal: .unit,
            n: 2, a: matrix, lda: 2, x: &x, incX: 1
        )
        expectApproximatelyEqual(x, [3, 10])
        BLAS.dtrsv(
            layout: .rowMajor, triangle: .lower,
            transpose: .noTranspose, diagonal: .unit,
            n: 2, a: matrix, lda: 2, x: &x, incX: 1
        )
        expectApproximatelyEqual(x, original)
    }

    @Test
    func triangularBandMultiplyAndSolveAreInverseOperations() {
        // Row-major upper-band storage for [[2, 1, 0], [0, 3, 4], [0, 0, 5]].
        let matrix = [2.0, 1, 3, 4, 5, 0]
        let original = [1.0, 2, 3]
        var x = original
        BLAS.dtbmv(
            layout: .rowMajor, triangle: .upper,
            transpose: .noTranspose, diagonal: .nonUnit,
            n: 3, k: 1, a: matrix, lda: 2,
            x: &x, incX: 1
        )
        expectApproximatelyEqual(x, [4, 18, 15])
        BLAS.dtbsv(
            layout: .rowMajor, triangle: .upper,
            transpose: .noTranspose, diagonal: .nonUnit,
            n: 3, k: 1, a: matrix, lda: 2,
            x: &x, incX: 1
        )
        expectApproximatelyEqual(x, original)
    }

    @Test
    func triangularPackedMultiplyAndSolveAreInverseOperations() {
        // Row-major lower-packed storage for [[2, 0, 0], [1, 3, 0], [4, 5, 6]].
        let packed = [2.0, 1, 3, 4, 5, 6]
        let original = [1.0, 2, 3]
        var x = original
        BLAS.dtpmv(
            layout: .rowMajor, triangle: .lower,
            transpose: .noTranspose, diagonal: .nonUnit,
            n: 3, ap: packed, x: &x, incX: 1
        )
        expectApproximatelyEqual(x, [2, 7, 32])
        BLAS.dtpsv(
            layout: .rowMajor, triangle: .lower,
            transpose: .noTranspose, diagonal: .nonUnit,
            n: 3, ap: packed, x: &x, incX: 1
        )
        expectApproximatelyEqual(x, original)
    }

    @Test
    func generalRankOneUpdatesDistinguishConjugation() {
        var realMatrix = [Double](repeating: 0, count: 6)
        BLAS.dger(
            layout: .rowMajor, m: 2, n: 3, alpha: 2,
            x: [1.0, 2], incX: 1,
            y: [3.0, 4, 5], incY: 1,
            a: &realMatrix, lda: 3
        )
        expectApproximatelyEqual(realMatrix, [6, 8, 10, 12, 16, 20])

        let complexX = [Complex<Double>(1, 1)]
        let complexY = [Complex<Double>(2, 1)]
        var unconjugated = [Complex<Double>.zero]
        var conjugated = [Complex<Double>.zero]
        BLAS.zgeru(
            layout: .rowMajor, m: 1, n: 1, alpha: .one,
            x: complexX, incX: 1, y: complexY, incY: 1,
            a: &unconjugated, lda: 1
        )
        BLAS.zgerc(
            layout: .rowMajor, m: 1, n: 1, alpha: .one,
            x: complexX, incX: 1, y: complexY, incY: 1,
            a: &conjugated, lda: 1
        )
        expectApproximatelyEqual(unconjugated[0], Complex(1, 3))
        expectApproximatelyEqual(conjugated[0], Complex(3, 1))
    }

    @Test
    func denseSymmetricAndHermitianRankUpdatesTouchOnlyOneTriangle() {
        var symmetric = [0.0, 0, 99, 0]
        BLAS.dsyr(
            layout: .rowMajor, triangle: .upper,
            n: 2, alpha: 2, x: [1.0, 2], incX: 1,
            a: &symmetric, lda: 2
        )
        expectApproximatelyEqual(symmetric, [2, 4, 99, 8])

        var symmetricRankTwo = [0.0, 0, 77, 0]
        BLAS.dsyr2(
            layout: .rowMajor, triangle: .upper,
            n: 2, alpha: 1,
            x: [1.0, 2], incX: 1,
            y: [3.0, 4], incY: 1,
            a: &symmetricRankTwo, lda: 2
        )
        expectApproximatelyEqual(symmetricRankTwo, [6, 10, 77, 16])

        var hermitian: [Complex<Double>] = [
            .zero, Complex(99, 99), .zero, .zero,
        ]
        BLAS.zher(
            layout: .rowMajor, triangle: .lower,
            n: 2, alpha: 2,
            x: [Complex(1, 1), Complex(2, -1)], incX: 1,
            a: &hermitian, lda: 2
        )
        expectApproximatelyEqual(hermitian[0], Complex(4, 0))
        expectApproximatelyEqual(hermitian[1], Complex(99, 99))
        expectApproximatelyEqual(hermitian[2], Complex(2, -6))
        expectApproximatelyEqual(hermitian[3], Complex(10, 0))

        var hermitianRankTwo = [Complex<Double>](repeating: .zero, count: 4)
        BLAS.zher2(
            layout: .rowMajor, triangle: .upper,
            n: 2, alpha: .one,
            x: [Complex(1, 0), .zero], incX: 1,
            y: [Complex(0, 1), .one], incY: 1,
            a: &hermitianRankTwo, lda: 2
        )
        expectApproximatelyEqual(hermitianRankTwo[0], .zero)
        expectApproximatelyEqual(hermitianRankTwo[1], .one)
        expectApproximatelyEqual(hermitianRankTwo[2], .zero)
        expectApproximatelyEqual(hermitianRankTwo[3], .zero)
    }

    @Test
    func packedSymmetricAndHermitianOperationsUseDocumentedOrdering() {
        let symmetricPacked = [1.0, 2, 3]
        var y = [Double](repeating: 0, count: 2)
        BLAS.dspmv(
            layout: .rowMajor, triangle: .upper,
            n: 2, alpha: 1, ap: symmetricPacked,
            x: [1.0, 2], incX: 1,
            beta: 0, y: &y, incY: 1
        )
        expectApproximatelyEqual(y, [5, 8])

        var symmetricRankOne = [Double](repeating: 0, count: 3)
        BLAS.dspr(
            layout: .rowMajor, triangle: .upper,
            n: 2, alpha: 2,
            x: [1.0, 2], incX: 1,
            ap: &symmetricRankOne
        )
        expectApproximatelyEqual(symmetricRankOne, [2, 4, 8])

        var symmetricRankTwo = [Double](repeating: 0, count: 3)
        BLAS.dspr2(
            layout: .rowMajor, triangle: .upper,
            n: 2, alpha: 1,
            x: [1.0, 2], incX: 1,
            y: [3.0, 4], incY: 1,
            ap: &symmetricRankTwo
        )
        expectApproximatelyEqual(symmetricRankTwo, [6, 10, 16])

        var hermitianRankOne = [Complex<Double>](repeating: .zero, count: 3)
        BLAS.zhpr(
            layout: .rowMajor, triangle: .lower,
            n: 2, alpha: 1,
            x: [Complex(1, 1), Complex(2, 0)], incX: 1,
            ap: &hermitianRankOne
        )
        expectApproximatelyEqual(
            hermitianRankOne,
            [Complex(2, 0), Complex(2, -2), Complex(4, 0)]
        )

        var hermitianRankTwo = [Complex<Double>](repeating: .zero, count: 3)
        BLAS.zhpr2(
            layout: .rowMajor, triangle: .upper,
            n: 2, alpha: .one,
            x: [Complex(1, 0), .zero], incX: 1,
            y: [Complex(0, 1), .one], incY: 1,
            ap: &hermitianRankTwo
        )
        expectApproximatelyEqual(
            hermitianRankTwo,
            [.zero, .one, .zero]
        )
    }
}
