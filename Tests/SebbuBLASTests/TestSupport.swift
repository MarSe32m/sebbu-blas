import ComplexModule
import RealModule
import Testing

func expectApproximatelyEqual(
    _ actual: [Double],
    _ expected: [Double],
    tolerance: Double = 1e-12
) {
    #expect(actual.count == expected.count)
    for (actual, expected) in zip(actual, expected) {
        #expect(
            actual.isApproximatelyEqual(
                to: expected,
                absoluteTolerance: tolerance
            )
        )
    }
}

func expectApproximatelyEqual(
    _ actual: [Float],
    _ expected: [Float],
    tolerance: Float = 1e-5
) {
    #expect(actual.count == expected.count)
    for (actual, expected) in zip(actual, expected) {
        #expect(
            actual.isApproximatelyEqual(
                to: expected,
                absoluteTolerance: tolerance
            )
        )
    }
}

func expectApproximatelyEqual(
    _ actual: Complex<Double>,
    _ expected: Complex<Double>,
    tolerance: Double = 1e-12
) {
    #expect(
        actual.real.isApproximatelyEqual(
            to: expected.real,
            absoluteTolerance: tolerance
        )
    )
    #expect(
        actual.imaginary.isApproximatelyEqual(
            to: expected.imaginary,
            absoluteTolerance: tolerance
        )
    )
}

func expectApproximatelyEqual(
    _ actual: Complex<Float>,
    _ expected: Complex<Float>,
    tolerance: Float = 1e-5
) {
    #expect(
        actual.real.isApproximatelyEqual(
            to: expected.real,
            absoluteTolerance: tolerance
        )
    )
    #expect(
        actual.imaginary.isApproximatelyEqual(
            to: expected.imaginary,
            absoluteTolerance: tolerance
        )
    )
}

func expectApproximatelyEqual(
    _ actual: [Complex<Double>],
    _ expected: [Complex<Double>],
    tolerance: Double = 1e-12
) {
    #expect(actual.count == expected.count)
    for (actual, expected) in zip(actual, expected) {
        expectApproximatelyEqual(actual, expected, tolerance: tolerance)
    }
}

func expectApproximatelyEqual(
    _ actual: [Complex<Float>],
    _ expected: [Complex<Float>],
    tolerance: Float = 1e-5
) {
    #expect(actual.count == expected.count)
    for (actual, expected) in zip(actual, expected) {
        expectApproximatelyEqual(actual, expected, tolerance: tolerance)
    }
}
