import ComplexModule
import RealModule
import SebbuBLAS
import Testing

@Suite("Level 1 BLAS")
struct Level1Tests {
    @Test
    func realAndMixedPrecisionDotProducts() {
        let x: [Float] = [1, 2, 3]
        let y: [Float] = [4, -5, 6]

        #expect(
            BLAS.sdsdot(n: 3, alpha: 1, x: x, incX: 1, y: y, incY: 1)
                .isApproximatelyEqual(to: 13, absoluteTolerance: 1e-5)
        )
        #expect(
            BLAS.dsdot(n: 3, x: x, incX: 1, y: y, incY: 1)
                .isApproximatelyEqual(to: 12, absoluteTolerance: 1e-12)
        )
        #expect(
            BLAS.sdot(n: 3, x: x, incX: 1, y: y, incY: 1)
                .isApproximatelyEqual(to: 12, absoluteTolerance: 1e-5)
        )

        let doubleX = x.map(Double.init)
        let doubleY = y.map(Double.init)
        #expect(
            BLAS.ddot(
                n: 3,
                x: doubleX, incX: 1,
                y: doubleY, incY: 1
            ).isApproximatelyEqual(to: 12, absoluteTolerance: 1e-12)
        )
    }

    @Test
    func complexDotProductsAndSubroutineSpellings() {
        let x: [Complex<Float>] = [
            Complex(1, 2), Complex(-3, 1),
        ]
        let y: [Complex<Float>] = [
            Complex(2, -1), Complex(4, 2),
        ]

        let unconjugated = BLAS.cdotu(
            n: 2, x: x, incX: 1, y: y, incY: 1
        )
        let conjugated = BLAS.cdotc(
            n: 2, x: x, incX: 1, y: y, incY: 1
        )
        expectApproximatelyEqual(unconjugated, Complex(-10, 1))
        expectApproximatelyEqual(conjugated, Complex(-10, -15))

        var subUnconjugated = Complex<Float>.zero
        var subConjugated = Complex<Float>.zero
        BLAS.cdotuSub(
            n: 2, x: x, incX: 1, y: y, incY: 1,
            result: &subUnconjugated
        )
        BLAS.cdotcSub(
            n: 2, x: x, incX: 1, y: y, incY: 1,
            result: &subConjugated
        )
        expectApproximatelyEqual(subUnconjugated, unconjugated)
        expectApproximatelyEqual(subConjugated, conjugated)

        let doubleX = x.map { Complex<Double>(Double($0.real), Double($0.imaginary)) }
        let doubleY = y.map { Complex<Double>(Double($0.real), Double($0.imaginary)) }
        var doubleResult = Complex<Double>.zero
        BLAS.zdotuSub(
            n: 2, x: doubleX, incX: 1,
            y: doubleY, incY: 1,
            result: &doubleResult
        )
        expectApproximatelyEqual(doubleResult, Complex(-10, 1))
    }

    @Test
    func sumsAbsoluteSumsAndNorms() {
        let real: [Double] = [-2, 3, -4]
        #expect(
            BLAS.dsum(n: 3, x: real, incX: 1)
                .isApproximatelyEqual(to: -3, absoluteTolerance: 1e-12)
        )
        #expect(
            BLAS.dasum(n: 3, x: real, incX: 1)
                .isApproximatelyEqual(to: 9, absoluteTolerance: 1e-12)
        )
        #expect(
            BLAS.dnrm2(n: 3, x: real, incX: 1)
                .isApproximatelyEqual(
                    to: 29.squareRoot(),
                    absoluteTolerance: 1e-12
                )
        )

        let complex: [Complex<Double>] = [Complex(3, 4), Complex(-1, 2)]
        #expect(
            BLAS.dzsum(n: 2, x: complex, incX: 1)
                .isApproximatelyEqual(to: 8, absoluteTolerance: 1e-12)
        )
        #expect(
            BLAS.dzasum(n: 2, x: complex, incX: 1)
                .isApproximatelyEqual(to: 10, absoluteTolerance: 1e-12)
        )
        #expect(
            BLAS.dznrm2(n: 2, x: complex, incX: 1)
                .isApproximatelyEqual(
                    to: 30.squareRoot(),
                    absoluteTolerance: 1e-12
                )
        )
    }

    @Test
    func axpyCopyAndSwapRespectNonUnitAndNegativeStrides() {
        let x = [1.0, 99, 2, 99, 3]
        var y = [10.0, 88, 20, 88, 30]
        BLAS.daxpy(n: 3, alpha: 2, x: x, incX: 2, y: &y, incY: 2)
        expectApproximatelyEqual(y, [12, 88, 24, 88, 36])

        var reversed = [0.0, 0, 0]
        x.withUnsafeBufferPointer { x in
            reversed.withUnsafeMutableBufferPointer { reversed in
                BLAS.dcopy(
                    n: 3,
                    x: x.baseAddress!, incX: -2,
                    y: reversed.baseAddress!, incY: 1
                )
            }
        }
        expectApproximatelyEqual(reversed, [3, 2, 1])

        var left = [1.0, 2, 3]
        var right = [4.0, 5, 6]
        BLAS.dswap(n: 3, x: &left, incX: 1, y: &right, incY: 1)
        expectApproximatelyEqual(left, [4, 5, 6])
        expectApproximatelyEqual(right, [1, 2, 3])
    }

    @Test
    func conjugatedAxpyAndComplexScaling() {
        let x: [Complex<Double>] = [Complex(1, 2), Complex(-3, 1)]
        var y: [Complex<Double>] = [Complex(2, -1), Complex(4, 2)]
        BLAS.zaxpyc(
            n: 2, alpha: Complex(2, 0),
            x: x, incX: 1, y: &y, incY: 1
        )
        expectApproximatelyEqual(
            y,
            [Complex(4, -5), Complex(-2, 0)]
        )

        BLAS.zscal(
            n: 2, alpha: Complex(0, 1),
            x: &y, incX: 1
        )
        expectApproximatelyEqual(
            y,
            [Complex(5, 4), Complex(0, -2)]
        )

        BLAS.zdscal(n: 2, alpha: 0.5, x: &y, incX: 1)
        expectApproximatelyEqual(
            y,
            [Complex(2.5, 2), Complex(0, -1)]
        )
    }

    @Test
    func planeRotationsAndRotationConstruction() {
        var x = [1.0, 2]
        var y = [3.0, 4]
        BLAS.drot(n: 2, x: &x, incX: 1, y: &y, incY: 1, c: 0, s: 1)
        expectApproximatelyEqual(x, [3, 4])
        expectApproximatelyEqual(y, [-1, -2])

        let originalA = 3.0
        let originalB = 4.0
        var a = originalA
        var b = originalB
        var c = 0.0
        var s = 0.0
        BLAS.drotg(a: &a, b: &b, c: &c, s: &s)
        #expect(
            (c * originalA + s * originalB)
                .isApproximatelyEqual(to: a, absoluteTolerance: 1e-12)
        )
        #expect(
            (c * originalB - s * originalA)
                .isApproximatelyEqual(to: 0, absoluteTolerance: 1e-12)
        )
        #expect(
            (c * c + s * s)
                .isApproximatelyEqual(to: 1, absoluteTolerance: 1e-12)
        )
    }

    @Test
    func modifiedPlaneRotationUsesAllCoefficients() {
        var x = [1.0, 2]
        var y = [3.0, 4]
        let parameters = [-1.0, 2, 3, 5, 7]
        BLAS.drotm(
            n: 2, x: &x, incX: 1,
            y: &y, incY: 1,
            parameters: parameters
        )
        expectApproximatelyEqual(x, [17, 24])
        expectApproximatelyEqual(y, [24, 34])
    }

    @Test
    func reductionExtensionsReturnLogicalIndicesAndValues() {
        let values = [-2.0, 3, -4]
        #expect(BLAS.idamax(n: 3, x: values, incX: 1) == 2)
        #expect(BLAS.idamin(n: 3, x: values, incX: 1) == 0)
        #expect(BLAS.idmax(n: 3, x: values, incX: 1) == 1)
        #expect(BLAS.idmin(n: 3, x: values, incX: 1) == 2)
        #expect(
            BLAS.damax(n: 3, x: values, incX: 1)
                .isApproximatelyEqual(to: 4, absoluteTolerance: 1e-12)
        )
        #expect(
            BLAS.damin(n: 3, x: values, incX: 1)
                .isApproximatelyEqual(to: 2, absoluteTolerance: 1e-12)
        )

        let complex: [Complex<Double>] = [
            Complex(1, 1), Complex(-4, 2), Complex(2, 1),
        ]
        #expect(BLAS.izamax(n: 3, x: complex, incX: 1) == 1)
        #expect(BLAS.izamin(n: 3, x: complex, incX: 1) == 0)
        #expect(
            BLAS.dzamax(n: 3, x: complex, incX: 1)
                .isApproximatelyEqual(to: 6, absoluteTolerance: 1e-12)
        )
        #expect(
            BLAS.dzamin(n: 3, x: complex, incX: 1)
                .isApproximatelyEqual(to: 2, absoluteTolerance: 1e-12)
        )
    }
}
