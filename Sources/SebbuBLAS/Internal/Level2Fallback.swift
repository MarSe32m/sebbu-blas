// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule

@inlinable
internal func _gemv<T: BLASScalar>(
    layout: BLAS.Layout,
    transpose: BLAS.Transpose,
    rows: BLAS.Index,
    columns: BLAS.Index,
    alpha: T,
    matrix: UnsafePointer<T>,
    leadingDimension: BLAS.Index,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    beta: T,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index
) {
    let m = _blasCount(rows)
    let n = _blasCount(columns)
    let outputCount = _isTransposed(transpose) ? n : m
    let innerCount = _isTransposed(transpose) ? m : n
    guard outputCount > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    let startX = _vectorStart(innerCount, incX)
    var iy = _vectorStart(outputCount, incY)
    for row in 0..<outputCount {
        var sum = T.zero
        if alpha != .zero {
            var ix = startX
            for column in 0..<innerCount {
                sum += _opElement(
                    matrix,
                    layout: layout,
                    transpose: transpose,
                    row: row,
                    column: column,
                    leadingDimension: leadingDimension
                ) * x[ix]
                ix &+= incX
            }
        }
        let product = innerCount == 0 || alpha == .zero ? T.zero : alpha * sum
        y[iy] = product + (beta == .zero ? .zero : beta * y[iy])
        iy += incY
    }
}

@inlinable
internal func _ger<T: BLASScalar>(
    layout: BLAS.Layout,
    rows: BLAS.Index,
    columns: BLAS.Index,
    alpha: T,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    y: UnsafePointer<T>,
    incY: BLAS.Index,
    matrix: UnsafeMutablePointer<T>,
    leadingDimension: BLAS.Index,
    conjugateY: Bool
) {
    let m = _blasCount(rows)
    let n = _blasCount(columns)
    guard m > 0 && n > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    let startX = _vectorStart(m, incX)
    let startY = _vectorStart(n, incY)
    var ix = startX
    for row in 0..<m {
        var iy = startY
        for column in 0..<n {
            let yValue = conjugateY ? y[iy].conjugate : y[iy]
            let index = _matrixIndex(layout, row, column, leadingDimension)
            matrix[index] += alpha * x[ix] * yValue
            iy += incY
        }
        ix += incX
    }
}

@inlinable
internal func _opTriangularElement<T: BLASScalar>(
    transpose: BLAS.Transpose,
    row: Int,
    column: Int,
    element: (Int, Int) -> T
) -> T {
    let sourceRow = _isTransposed(transpose) ? column : row
    let sourceColumn = _isTransposed(transpose) ? row : column
    let value = element(sourceRow, sourceColumn)
    return _isConjugated(transpose) ? value.conjugate : value
}

@inlinable
internal func _triangularVectorMultiply<T: BLASScalar>(
    triangle: BLAS.Triangle,
    transpose: BLAS.Transpose,
    order: BLAS.Index,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index,
    element: (Int, Int) -> T
) {
    let n = _blasCount(order)
    guard n > 0 else { return }
    precondition(incX != 0, "BLAS vector strides must be nonzero")
    let start = _vectorStart(n, incX)
    var source = Array(repeating: T.zero, count: n)
    var ix = start
    for i in 0..<n { source[i] = x[ix]; ix += incX }
    ix = start
    for row in 0..<n {
        var sum = T.zero
        for column in 0..<n {
            sum += _opTriangularElement(
                transpose: transpose,
                row: row,
                column: column,
                element: element
            ) * source[column]
        }
        x[ix] = sum
        ix += incX
    }
}

@inlinable
internal func _triangularVectorSolve<T: BLASScalar>(
    triangle: BLAS.Triangle,
    transpose: BLAS.Transpose,
    order: BLAS.Index,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index,
    element: (Int, Int) -> T
) {
    let n = _blasCount(order)
    guard n > 0 else { return }
    precondition(incX != 0, "BLAS vector strides must be nonzero")
    let start = _vectorStart(n, incX)
    let effectiveUpper = _isTransposed(transpose) ? triangle == .lower : triangle == .upper
    if effectiveUpper {
        for row in stride(from: n - 1, through: 0, by: -1) {
            let rowIndex = start + row * incX
            var value = x[rowIndex]
            if row + 1 < n {
                for column in (row + 1)..<n {
                    value -= _opTriangularElement(
                        transpose: transpose,
                        row: row,
                        column: column,
                        element: element
                    ) * x[start + column * incX]
                }
            }
            value = value / _opTriangularElement(
                transpose: transpose,
                row: row,
                column: row,
                element: element
            )
            x[rowIndex] = value
        }
    } else {
        for row in 0..<n {
            let rowIndex = start + row * incX
            var value = x[rowIndex]
            if row > 0 {
                for column in 0..<row {
                    value -= _opTriangularElement(
                        transpose: transpose,
                        row: row,
                        column: column,
                        element: element
                    ) * x[start + column * incX]
                }
            }
            value = value / _opTriangularElement(
                transpose: transpose,
                row: row,
                column: row,
                element: element
            )
            x[rowIndex] = value
        }
    }
}

@inlinable
internal func _trmv<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    transpose: BLAS.Transpose,
    diagonal: BLAS.Diagonal,
    order: BLAS.Index,
    matrix: UnsafePointer<T>,
    leadingDimension: BLAS.Index,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index
) {
    _triangularVectorMultiply(
        triangle: triangle,
        transpose: transpose,
        order: order,
        x: x,
        incX: incX
    ) { row, column in
        _triangularElement(
            matrix, layout: layout, triangle: triangle, diagonal: diagonal,
            row: row, column: column, leadingDimension: leadingDimension
        )
    }
}

@inlinable
internal func _trsv<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    transpose: BLAS.Transpose,
    diagonal: BLAS.Diagonal,
    order: BLAS.Index,
    matrix: UnsafePointer<T>,
    leadingDimension: BLAS.Index,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index
) {
    _triangularVectorSolve(
        triangle: triangle,
        transpose: transpose,
        order: order,
        x: x,
        incX: incX
    ) { row, column in
        _triangularElement(
            matrix, layout: layout, triangle: triangle, diagonal: diagonal,
            row: row, column: column, leadingDimension: leadingDimension
        )
    }
}

@inlinable
internal func _rank1Triangle<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    order: BLAS.Index,
    alpha: T,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    matrix: UnsafeMutablePointer<T>,
    leadingDimension: BLAS.Index,
    hermitian: Bool
) {
    let n = _blasCount(order)
    guard n > 0 else { return }
    precondition(incX != 0, "BLAS vector strides must be nonzero")
    let start = _vectorStart(n, incX)
    for row in 0..<n {
        for column in 0..<n where (triangle == .upper ? row <= column : row >= column) {
            let lhs = x[start + row * incX]
            let rhs = x[start + column * incX]
            let index = _matrixIndex(layout, row, column, leadingDimension)
            matrix[index] += alpha * lhs * (hermitian ? rhs.conjugate : rhs)
            if hermitian && row == column { matrix[index] = matrix[index].withZeroImaginary }
        }
    }
}

@inlinable
internal func _rank2Triangle<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    order: BLAS.Index,
    alpha: T,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    y: UnsafePointer<T>,
    incY: BLAS.Index,
    matrix: UnsafeMutablePointer<T>,
    leadingDimension: BLAS.Index,
    hermitian: Bool
) {
    let n = _blasCount(order)
    guard n > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    let startX = _vectorStart(n, incX)
    let startY = _vectorStart(n, incY)
    for row in 0..<n {
        for column in 0..<n where (triangle == .upper ? row <= column : row >= column) {
            let xr = x[startX + row * incX]
            let xc = x[startX + column * incX]
            let yr = y[startY + row * incY]
            let yc = y[startY + column * incY]
            let update: T
            if hermitian {
                update = alpha * xr * yc.conjugate + alpha.conjugate * yr * xc.conjugate
            } else {
                update = alpha * (xr * yc + yr * xc)
            }
            let index = _matrixIndex(layout, row, column, leadingDimension)
            matrix[index] += update
            if hermitian && row == column { matrix[index] = matrix[index].withZeroImaginary }
        }
    }
}

@inlinable
internal func _gbmv<T: BLASScalar>(
    layout: BLAS.Layout,
    transpose: BLAS.Transpose,
    rows: BLAS.Index,
    columns: BLAS.Index,
    lowerBandwidth: BLAS.Index,
    upperBandwidth: BLAS.Index,
    alpha: T,
    matrix: UnsafePointer<T>,
    leadingDimension: BLAS.Index,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    beta: T,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index
) {
    let m = _blasCount(rows)
    let n = _blasCount(columns)
    let kl = _blasCount(lowerBandwidth)
    let ku = _blasCount(upperBandwidth)
    let outputCount = _isTransposed(transpose) ? n : m
    let innerCount = _isTransposed(transpose) ? m : n
    guard outputCount > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    let startX = _vectorStart(innerCount, incX)
    var iy = _vectorStart(outputCount, incY)
    for row in 0..<outputCount {
        var sum = T.zero
        if alpha != .zero {
            var ix = startX
            for column in 0..<innerCount {
                let sourceRow = _isTransposed(transpose) ? column : row
                let sourceColumn = _isTransposed(transpose) ? row : column
                var value = _generalBandElement(
                    matrix, layout: layout, rows: m, columns: n,
                    lowerBandwidth: kl, upperBandwidth: ku,
                    row: sourceRow, column: sourceColumn,
                    leadingDimension: leadingDimension
                )
                if _isConjugated(transpose) { value = value.conjugate }
                sum += value * x[ix]
                ix += incX
            }
        }
        let product = innerCount == 0 || alpha == .zero ? T.zero : alpha * sum
        y[iy] = product + (beta == .zero ? .zero : beta * y[iy])
        iy += incY
    }
}

@inlinable
internal func _structuredMV<T: BLASScalar>(
    order: BLAS.Index,
    alpha: T,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    beta: T,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index,
    element: (Int, Int) -> T
) {
    let n = _blasCount(order)
    guard n > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    let startX = _vectorStart(n, incX)
    var iy = _vectorStart(n, incY)
    for row in 0..<n {
        var sum = T.zero
        if alpha != .zero {
            var ix = startX
            for column in 0..<n {
                sum += element(row, column) * x[ix]
                ix += incX
            }
        }
        y[iy] = alpha * sum + (beta == .zero ? .zero : beta * y[iy])
        iy += incY
    }
}

@inlinable
internal func _symv<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    order: BLAS.Index,
    alpha: T,
    matrix: UnsafePointer<T>,
    leadingDimension: BLAS.Index,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    beta: T,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index,
    hermitian: Bool
) {
    _structuredMV(order: order, alpha: alpha, x: x, incX: incX, beta: beta, y: y, incY: incY) {
        _symmetricElement(
            matrix, layout: layout, triangle: triangle,
            row: $0, column: $1, leadingDimension: leadingDimension,
            hermitian: hermitian
        )
    }
}

@inlinable
internal func _sbmv<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    order: BLAS.Index,
    bandwidth: BLAS.Index,
    alpha: T,
    matrix: UnsafePointer<T>,
    leadingDimension: BLAS.Index,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    beta: T,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index,
    hermitian: Bool
) {
    let n = _blasCount(order)
    let k = _blasCount(bandwidth)
    _structuredMV(order: order, alpha: alpha, x: x, incX: incX, beta: beta, y: y, incY: incY) {
        _symmetricBandElement(
            matrix, layout: layout, triangle: triangle, order: n,
            bandwidth: k, row: $0, column: $1,
            leadingDimension: leadingDimension, hermitian: hermitian
        )
    }
}

@inlinable
internal func _spmv<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    order: BLAS.Index,
    alpha: T,
    packedMatrix: UnsafePointer<T>,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    beta: T,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index,
    hermitian: Bool
) {
    let n = _blasCount(order)
    _structuredMV(order: order, alpha: alpha, x: x, incX: incX, beta: beta, y: y, incY: incY) {
        _packedElement(
            packedMatrix, layout: layout, triangle: triangle, order: n,
            row: $0, column: $1, hermitian: hermitian
        )
    }
}

@inlinable
internal func _tbmv<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    transpose: BLAS.Transpose,
    diagonal: BLAS.Diagonal,
    order: BLAS.Index,
    bandwidth: BLAS.Index,
    matrix: UnsafePointer<T>,
    leadingDimension: BLAS.Index,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index,
    solve: Bool
) {
    let n = _blasCount(order)
    let k = _blasCount(bandwidth)
    if solve {
        _triangularVectorSolve(triangle: triangle, transpose: transpose, order: order, x: x, incX: incX) { row, column in
            _triangularBandElement(
                matrix, layout: layout, triangle: triangle, diagonal: diagonal,
                order: n, bandwidth: k, row: row, column: column,
                leadingDimension: leadingDimension
            )
        }
    } else {
        _triangularVectorMultiply(triangle: triangle, transpose: transpose, order: order, x: x, incX: incX) { row, column in
            _triangularBandElement(
                matrix, layout: layout, triangle: triangle, diagonal: diagonal,
                order: n, bandwidth: k, row: row, column: column,
                leadingDimension: leadingDimension
            )
        }
    }
}

@inlinable
internal func _tpmv<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    transpose: BLAS.Transpose,
    diagonal: BLAS.Diagonal,
    order: BLAS.Index,
    packedMatrix: UnsafePointer<T>,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index,
    solve: Bool
) {
    let n = _blasCount(order)
    if solve {
        _triangularVectorSolve(triangle: triangle, transpose: transpose, order: order, x: x, incX: incX) { row, column in
            _triangularPackedElement(
                packedMatrix, layout: layout, triangle: triangle, diagonal: diagonal,
                order: n, row: row, column: column
            )
        }
    } else {
        _triangularVectorMultiply(triangle: triangle, transpose: transpose, order: order, x: x, incX: incX) { row, column in
            _triangularPackedElement(
                packedMatrix, layout: layout, triangle: triangle, diagonal: diagonal,
                order: n, row: row, column: column
            )
        }
    }
}

@inlinable
internal func _packedRank1<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    order: BLAS.Index,
    alpha: T,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    packedMatrix: UnsafeMutablePointer<T>,
    hermitian: Bool
) {
    let n = _blasCount(order)
    guard n > 0 else { return }
    precondition(incX != 0, "BLAS vector strides must be nonzero")
    let start = _vectorStart(n, incX)
    for row in 0..<n {
        for column in 0..<n where (triangle == .upper ? row <= column : row >= column) {
            let index = _packedIndex(layout: layout, triangle: triangle, order: n, row: row, column: column)
            let rhs = x[start + column * incX]
            packedMatrix[index] += alpha * x[start + row * incX] * (hermitian ? rhs.conjugate : rhs)
            if hermitian && row == column { packedMatrix[index] = packedMatrix[index].withZeroImaginary }
        }
    }
}

@inlinable
internal func _packedRank2<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    order: BLAS.Index,
    alpha: T,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    y: UnsafePointer<T>,
    incY: BLAS.Index,
    packedMatrix: UnsafeMutablePointer<T>,
    hermitian: Bool
) {
    let n = _blasCount(order)
    guard n > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    let startX = _vectorStart(n, incX)
    let startY = _vectorStart(n, incY)
    for row in 0..<n {
        for column in 0..<n where (triangle == .upper ? row <= column : row >= column) {
            let xr = x[startX + row * incX]
            let xc = x[startX + column * incX]
            let yr = y[startY + row * incY]
            let yc = y[startY + column * incY]
            let update = hermitian
                ? alpha * xr * yc.conjugate + alpha.conjugate * yr * xc.conjugate
                : alpha * (xr * yc + yr * xc)
            let index = _packedIndex(layout: layout, triangle: triangle, order: n, row: row, column: column)
            packedMatrix[index] += update
            if hermitian && row == column { packedMatrix[index] = packedMatrix[index].withZeroImaginary }
        }
    }
}
