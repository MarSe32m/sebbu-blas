import ComplexModule
import RealModule
import SebbuBLAS
import Testing

@Suite("Level 3 BLAS")
struct Level3Tests {
    @Test
    func complexGemmSupportsConjugateTransposeAndBeta() {
        let a: [Complex<Double>] = [
            Complex(1, 1), Complex(2, 0),
            Complex(3, -1), Complex(4, 2),
        ]
        let identity: [Complex<Double>] = [
            .one, .zero, .zero, .one,
        ]
        var c = [Complex<Double>](repeating: .one, count: 4)
        BLAS.zgemm(
            layout: .rowMajor,
            transposeA: .conjugateTranspose,
            transposeB: .noTranspose,
            m: 2, n: 2, k: 2,
            alpha: .one, a: a, lda: 2,
            b: identity, ldb: 2,
            beta: Complex(2, 0), c: &c, ldc: 2
        )
        expectApproximatelyEqual(
            c,
            [Complex(3, -1), Complex(5, 1), Complex(4, 0), Complex(6, -2)]
        )
    }

    @Test
    func singlePrecisionGemmEntryPoints() {
        var real = [Float.zero]
        BLAS.sgemm(
            layout: .columnMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 1, n: 1, k: 1,
            alpha: 2, a: [3], lda: 1,
            b: [4], ldb: 1,
            beta: 0, c: &real, ldc: 1
        )
        expectApproximatelyEqual(real, [24])

        var complex = [Complex<Float>.zero]
        BLAS.cgemm(
            layout: .rowMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 1, n: 1, k: 1,
            alpha: .one, a: [Complex<Float>(1, 1)], lda: 1,
            b: [Complex<Float>(2, -1)], ldb: 1,
            beta: .zero, c: &complex, ldc: 1
        )
        expectApproximatelyEqual(complex, [Complex(3, 1)])
    }

    @Test
    func symmetricAndHermitianMatrixMultiplication() {
        // Only the lower triangle stores symmetric [[1, 2], [2, 3]].
        let symmetric = [1.0, 99, 2, 3]
        let b = [1.0, 2, 3, 4]
        var symmetricResult = [Double](repeating: 0, count: 4)
        BLAS.dsymm(
            layout: .rowMajor, side: .right, triangle: .lower,
            m: 2, n: 2, alpha: 1,
            a: symmetric, lda: 2,
            b: b, ldb: 2,
            beta: 0, c: &symmetricResult, ldc: 2
        )
        expectApproximatelyEqual(symmetricResult, [5, 8, 11, 18])

        let hermitian: [Complex<Double>] = [
            Complex(1, 9), Complex(2, 1),
            Complex(99, 99), Complex(3, 8),
        ]
        let identity: [Complex<Double>] = [.one, .zero, .zero, .one]
        var result = [Complex<Double>](repeating: .zero, count: 4)
        BLAS.zhemm(
            layout: .rowMajor, side: .left, triangle: .upper,
            m: 2, n: 2, alpha: .one,
            a: hermitian, lda: 2,
            b: identity, ldb: 2,
            beta: .zero, c: &result, ldc: 2
        )
        expectApproximatelyEqual(
            result,
            [Complex(1, 0), Complex(2, 1), Complex(2, -1), Complex(3, 0)]
        )
    }

    @Test
    func symmetricAndHermitianRankKUpdatesPreserveUnusedTriangle() {
        let a = [1.0, 2, 3, 4, 5, 6]
        var symmetric = [0.0, 0, 99, 0]
        BLAS.dsyrk(
            layout: .rowMajor, triangle: .upper,
            transpose: .noTranspose,
            n: 2, k: 3, alpha: 1,
            a: a, lda: 3,
            beta: 0, c: &symmetric, ldc: 2
        )
        expectApproximatelyEqual(symmetric, [14, 32, 99, 77])

        let complexA: [Complex<Double>] = [Complex(1, 1), Complex(2, -1)]
        var hermitian: [Complex<Double>] = [
            .zero, .zero, Complex(99, 99), .zero,
        ]
        BLAS.zherk(
            layout: .rowMajor, triangle: .upper,
            transpose: .noTranspose,
            n: 2, k: 1, alpha: 1,
            a: complexA, lda: 1,
            beta: 0, c: &hermitian, ldc: 2
        )
        expectApproximatelyEqual(hermitian[0], Complex(2, 0))
        expectApproximatelyEqual(hermitian[1], Complex(1, 3))
        expectApproximatelyEqual(hermitian[2], Complex(99, 99))
        expectApproximatelyEqual(hermitian[3], Complex(5, 0))
    }

    @Test
    func symmetricAndHermitianRankTwoKUpdates() {
        var symmetric = [0.0, 0, 99, 0]
        BLAS.dsyr2k(
            layout: .rowMajor, triangle: .upper,
            transpose: .noTranspose,
            n: 2, k: 1, alpha: 1,
            a: [1.0, 2], lda: 1,
            b: [3.0, 4], ldb: 1,
            beta: 0, c: &symmetric, ldc: 2
        )
        expectApproximatelyEqual(symmetric, [6, 10, 99, 16])

        var hermitian = [Complex<Double>](repeating: .zero, count: 4)
        BLAS.zher2k(
            layout: .rowMajor, triangle: .upper,
            transpose: .noTranspose,
            n: 2, k: 1, alpha: .one,
            a: [Complex(1, 0), .zero], lda: 1,
            b: [Complex(0, 1), .one], ldb: 1,
            beta: 0, c: &hermitian, ldc: 2
        )
        expectApproximatelyEqual(
            hermitian,
            [.zero, .one, .zero, .zero]
        )
    }

    @Test
    func triangularGemmtUpdatesOnlyRequestedTriangle() {
        let a = [1.0, 2, 3, 4]
        let identity = [1.0, 0, 0, 1]
        var c = [10.0, 99, 20, 30]
        BLAS.dgemmt(
            layout: .rowMajor, triangle: .lower,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 2, k: 2, alpha: 2,
            a: a, lda: 2, b: identity, ldb: 2,
            beta: 0.5, c: &c, ldc: 2
        )
        expectApproximatelyEqual(c, [7, 99, 16, 23])
    }

    @Test
    func leftTriangularMultiplyAndSolveAreInverseOperations() {
        let triangular = [2.0, 1, 0, 3]
        let original = [1.0, 2, 3, 4]
        var b = original
        BLAS.dtrmm(
            layout: .rowMajor, side: .left, triangle: .upper,
            transpose: .noTranspose, diagonal: .nonUnit,
            m: 2, n: 2, alpha: 1,
            a: triangular, lda: 2,
            b: &b, ldb: 2
        )
        expectApproximatelyEqual(b, [5, 8, 9, 12])
        BLAS.dtrsm(
            layout: .rowMajor, side: .left, triangle: .upper,
            transpose: .noTranspose, diagonal: .nonUnit,
            m: 2, n: 2, alpha: 1,
            a: triangular, lda: 2,
            b: &b, ldb: 2
        )
        expectApproximatelyEqual(b, original)
    }

    @Test
    func rightUnitTriangularMultiplyAndSolveAreInverseOperations() {
        let triangular: [Float] = [99, 0, 2, -17]
        let original: [Float] = [1, 2, 3, 4]
        var b = original
        BLAS.strmm(
            layout: .rowMajor, side: .right, triangle: .lower,
            transpose: .noTranspose, diagonal: .unit,
            m: 2, n: 2, alpha: 1,
            a: triangular, lda: 2,
            b: &b, ldb: 2
        )
        expectApproximatelyEqual(b, [5, 2, 11, 4])
        BLAS.strsm(
            layout: .rowMajor, side: .right, triangle: .lower,
            transpose: .noTranspose, diagonal: .unit,
            m: 2, n: 2, alpha: 1,
            a: triangular, lda: 2,
            b: &b, ldb: 2
        )
        expectApproximatelyEqual(b, original)
    }
}
