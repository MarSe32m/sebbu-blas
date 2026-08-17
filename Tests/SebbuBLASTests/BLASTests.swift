import ComplexModule
import RealModule
import Testing
@testable import SebbuBLAS

@Suite("BLAS tests")
struct BLASTests {
    @Test
    func testLevelOneRealAndNegativeStrides() {
        let x = [1.0, 2.0, 3.0]
        let y = [4.0, 5.0, 6.0]
        let value = x.withUnsafeBufferPointer { x in
            y.withUnsafeBufferPointer { y in
                BLAS.ddot(n: 3, x: x.baseAddress!, incX: -1, y: y.baseAddress!, incY: 1)
            }
        }
        #expect(value.isApproximatelyEqual(to: 28, absoluteTolerance: 1e-12))

        let index: Int = x.withUnsafeBufferPointer { x in
            BLAS.idamax(n: x.count, x: x.baseAddress!, incX: 1)
        }
        #expect(index == 2)
    }

    @Test
    func testComplexConjugatedDotProduct() {
        let x = [Complex<Double>(1, 2), Complex<Double>(3, -1)]
        let y = [Complex<Double>(2, -1), Complex<Double>(-1, 4)]
        let value = x.withUnsafeBufferPointer { x in
            y.withUnsafeBufferPointer { y in
                BLAS.zdotc(n: 2, x: x.baseAddress!, incX: 1, y: y.baseAddress!, incY: 1)
            }
        }
        #expect(value.real.isApproximatelyEqual(to: -7, absoluteTolerance: 1e-12))
        #expect(value.imaginary.isApproximatelyEqual(to: 6, absoluteTolerance: 1e-12))
    }

    @Test
    func testRowMajorMatrixMultiplication() {
        let a = [1.0, 2, 3, 4, 5, 6]
        let b = [7.0, 8, 9, 10, 11, 12]
        var c = Array(repeating: 0.0, count: 4)
        a.withUnsafeBufferPointer { a in
            b.withUnsafeBufferPointer { b in
                c.withUnsafeMutableBufferPointer { c in
                    BLAS.dgemm(
                        layout: .rowMajor,
                        transposeA: .noTranspose,
                        transposeB: .noTranspose,
                        m: 2, n: 2, k: 3,
                        alpha: 1,
                        a: a.baseAddress!, lda: 3,
                        b: b.baseAddress!, ldb: 2,
                        beta: 0,
                        c: c.baseAddress!, ldc: 2
                    )
                }
            }
        }
        #expect(c[0].isApproximatelyEqual(to: 58, absoluteTolerance: 1e-12))
        #expect(c[1].isApproximatelyEqual(to: 64, absoluteTolerance: 1e-12))
        #expect(c[2].isApproximatelyEqual(to: 139, absoluteTolerance: 1e-12))
        #expect(c[3].isApproximatelyEqual(to: 154, absoluteTolerance: 1e-12))
    }

    @Test
    func testTriangularSolve() {
        let a = [2.0, 1, 0, 3]
        var x = [5.0, 6]
        BLAS.dtrsv(
            layout: .rowMajor,
            triangle: .upper,
            transpose: .noTranspose,
            diagonal: .nonUnit,
            n: 2,
            a: a, lda: 2,
            x: &x, incX: 1
        )
        #expect(x[0].isApproximatelyEqual(to: 1.5, absoluteTolerance: 1e-12))
        #expect(x[1].isApproximatelyEqual(to: 2, absoluteTolerance: 1e-12))
    }

    @Test
    func testTriangularBandDoesNotReflectTheStoredTriangle() {
        let a = [1.0, 2, 3, 4, 5, 0]
        var x = [1.0, 1, 1]
        BLAS.dtbmv(
            layout: .rowMajor,
            triangle: .upper,
            transpose: .noTranspose,
            diagonal: .nonUnit,
            n: 3, k: 1,
            a: a, lda: 2,
            x: &x, incX: 1
        )
        #expect(x == [3, 7, 5])
    }

    @Test
    func testHermitianDiagonalIgnoresImaginaryComponent() {
        let a = [
            Complex<Double>(1, 9), Complex<Double>(2, 3),
            Complex<Double>.zero, Complex<Double>(4, 8),
        ]
        let x = [Complex<Double>(1, 0), .zero]
        var y = [Complex<Double>.zero, .zero]
        BLAS.zhemv(
            layout: .rowMajor, triangle: .upper, n: 2,
            alpha: Complex<Double>(1, 0),
            a: a, lda: 2,
            x: x, incX: 1,
            beta: .zero,
            y: &y, incY: 1
        )
        #expect(y[0].real.isApproximatelyEqual(to: 1, absoluteTolerance: 1e-12))
        #expect(y[0].imaginary.isApproximatelyEqual(to: 0, absoluteTolerance: 1e-12))
        #expect(y[1].real.isApproximatelyEqual(to: 2, absoluteTolerance: 1e-12))
        #expect(y[1].imaginary.isApproximatelyEqual(to: -3, absoluteTolerance: 1e-12))
    }

    @Test
    func testOutOfPlaceTransposeExtension() {
        let a: [Float] = [1, 2, 3, 4, 5, 6]
        var b = Array(repeating: Float.zero, count: 6)
        BLAS.somatcopy(
            layout: .rowMajor,
            transpose: .transpose,
            rows: 2, columns: 3,
            alpha: 1,
            a: a, lda: 3,
            b: &b, ldb: 2
        )
        #expect(b == [1, 4, 2, 5, 3, 6])
    }

    @Test
    func testColumnMajorInPlaceTransposeExtension() {
        var matrix = [1.0, 4, 2, 5, 3, 6]
        BLAS.dimatcopy(
            layout: .columnMajor,
            transpose: .transpose,
            rows: 2, columns: 3,
            alpha: 1,
            a: &matrix, lda: 2, ldb: 3
        )
        #expect(matrix == [1, 2, 3, 4, 5, 6])
    }
}
