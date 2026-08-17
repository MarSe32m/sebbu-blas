// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule

@inlinable
internal func _gemm<T: BLASScalar>(
    layout: BLAS.Layout,
    transposeA: BLAS.Transpose,
    transposeB: BLAS.Transpose,
    m: BLAS.Index,
    n: BLAS.Index,
    k: BLAS.Index,
    alpha: T,
    a: UnsafePointer<T>,
    lda: BLAS.Index,
    b: UnsafePointer<T>,
    ldb: BLAS.Index,
    beta: T,
    c: UnsafeMutablePointer<T>,
    ldc: BLAS.Index
) {
    let rows = _blasCount(m)
    let columns = _blasCount(n)
    let inner = _blasCount(k)
    guard rows > 0 && columns > 0 else { return }

    for row in 0..<rows {
        for column in 0..<columns {
            let index = _matrixIndex(layout, row, column, ldc)
            c[index] = beta == .zero ? .zero : beta * c[index]
        }
    }
    guard inner > 0 && alpha != .zero else { return }

    let block = 32
    var kk = 0
    while kk < inner {
        let kEnd = Swift.min(kk + block, inner)
        var ii = 0
        while ii < rows {
            let iEnd = Swift.min(ii + block, rows)
            var jj = 0
            while jj < columns {
                let jEnd = Swift.min(jj + block, columns)
                for row in ii..<iEnd {
                    for column in jj..<jEnd {
                        var sum = T.zero
                        for innerIndex in kk..<kEnd {
                            sum += _opElement(
                                a, layout: layout, transpose: transposeA,
                                row: row, column: innerIndex,
                                leadingDimension: lda
                            ) * _opElement(
                                b, layout: layout, transpose: transposeB,
                                row: innerIndex, column: column,
                                leadingDimension: ldb
                            )
                        }
                        c[_matrixIndex(layout, row, column, ldc)] += alpha * sum
                    }
                }
                jj &+= block
            }
            ii &+= block
        }
        kk &+= block
    }
}

@inlinable
internal func _gemmt<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    transposeA: BLAS.Transpose,
    transposeB: BLAS.Transpose,
    m: BLAS.Index,
    k: BLAS.Index,
    alpha: T,
    a: UnsafePointer<T>,
    lda: BLAS.Index,
    b: UnsafePointer<T>,
    ldb: BLAS.Index,
    beta: T,
    c: UnsafeMutablePointer<T>,
    ldc: BLAS.Index
) {
    let size = _blasCount(m)
    let inner = _blasCount(k)
    for row in 0..<size {
        for column in 0..<size where (triangle == .upper ? row <= column : row >= column) {
            var sum = T.zero
            if alpha != .zero {
                for p in 0..<inner {
                    sum += _opElement(a, layout: layout, transpose: transposeA, row: row, column: p, leadingDimension: lda)
                        * _opElement(b, layout: layout, transpose: transposeB, row: p, column: column, leadingDimension: ldb)
                }
            }
            let index = _matrixIndex(layout, row, column, ldc)
            let product = inner == 0 || alpha == .zero ? T.zero : alpha * sum
            c[index] = product + (beta == .zero ? .zero : beta * c[index])
        }
    }
}

@inlinable
internal func _symm<T: BLASScalar>(
    layout: BLAS.Layout,
    side: BLAS.Side,
    triangle: BLAS.Triangle,
    m: BLAS.Index,
    n: BLAS.Index,
    alpha: T,
    a: UnsafePointer<T>,
    lda: BLAS.Index,
    b: UnsafePointer<T>,
    ldb: BLAS.Index,
    beta: T,
    c: UnsafeMutablePointer<T>,
    ldc: BLAS.Index,
    hermitian: Bool
) {
    let rows = _blasCount(m)
    let columns = _blasCount(n)
    let inner = side == .left ? rows : columns
    for row in 0..<rows {
        for column in 0..<columns {
            var sum = T.zero
            if alpha != .zero {
                for p in 0..<inner {
                    if side == .left {
                        sum += _symmetricElement(
                            a, layout: layout, triangle: triangle,
                            row: row, column: p, leadingDimension: lda,
                            hermitian: hermitian
                        ) * b[_matrixIndex(layout, p, column, ldb)]
                    } else {
                        sum += b[_matrixIndex(layout, row, p, ldb)] * _symmetricElement(
                            a, layout: layout, triangle: triangle,
                            row: p, column: column, leadingDimension: lda,
                            hermitian: hermitian
                        )
                    }
                }
            }
            let index = _matrixIndex(layout, row, column, ldc)
            let product = inner == 0 || alpha == .zero ? T.zero : alpha * sum
            c[index] = product + (beta == .zero ? .zero : beta * c[index])
        }
    }
}

@inlinable
internal func _rankK<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    transpose: BLAS.Transpose,
    n: BLAS.Index,
    k: BLAS.Index,
    alpha: T,
    a: UnsafePointer<T>,
    lda: BLAS.Index,
    beta: T,
    c: UnsafeMutablePointer<T>,
    ldc: BLAS.Index,
    hermitian: Bool
) {
    let size = _blasCount(n)
    let inner = _blasCount(k)
    for row in 0..<size {
        for column in 0..<size where (triangle == .upper ? row <= column : row >= column) {
            var sum = T.zero
            if alpha != .zero {
                for p in 0..<inner {
                    let lhs = _opElement(a, layout: layout, transpose: transpose, row: row, column: p, leadingDimension: lda)
                    var rhs = _opElement(a, layout: layout, transpose: transpose, row: column, column: p, leadingDimension: lda)
                    if hermitian { rhs = rhs.conjugate }
                    sum += lhs * rhs
                }
            }
            let index = _matrixIndex(layout, row, column, ldc)
            let product = inner == 0 || alpha == .zero ? T.zero : alpha * sum
            c[index] = product + (beta == .zero ? .zero : beta * c[index])
            if hermitian && row == column { c[index] = c[index].withZeroImaginary }
        }
    }
}

@inlinable
internal func _rank2K<T: BLASScalar>(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    transpose: BLAS.Transpose,
    n: BLAS.Index,
    k: BLAS.Index,
    alpha: T,
    a: UnsafePointer<T>,
    lda: BLAS.Index,
    b: UnsafePointer<T>,
    ldb: BLAS.Index,
    beta: T,
    c: UnsafeMutablePointer<T>,
    ldc: BLAS.Index,
    hermitian: Bool
) {
    let size = _blasCount(n)
    let inner = _blasCount(k)
    for row in 0..<size {
        for column in 0..<size where (triangle == .upper ? row <= column : row >= column) {
            var sum = T.zero
            if alpha != .zero {
                for p in 0..<inner {
                    let ar = _opElement(a, layout: layout, transpose: transpose, row: row, column: p, leadingDimension: lda)
                    let ac = _opElement(a, layout: layout, transpose: transpose, row: column, column: p, leadingDimension: lda)
                    let br = _opElement(b, layout: layout, transpose: transpose, row: row, column: p, leadingDimension: ldb)
                    let bc = _opElement(b, layout: layout, transpose: transpose, row: column, column: p, leadingDimension: ldb)
                    if hermitian {
                        sum += alpha * ar * bc.conjugate + alpha.conjugate * br * ac.conjugate
                    } else {
                        sum += alpha * (ar * bc + br * ac)
                    }
                }
            }
            let index = _matrixIndex(layout, row, column, ldc)
            c[index] = sum + (beta == .zero ? .zero : beta * c[index])
            if hermitian && row == column { c[index] = c[index].withZeroImaginary }
        }
    }
}

@inlinable
internal func _trmm<T: BLASScalar>(
    layout: BLAS.Layout,
    side: BLAS.Side,
    triangle: BLAS.Triangle,
    transpose: BLAS.Transpose,
    diagonal: BLAS.Diagonal,
    m: BLAS.Index,
    n: BLAS.Index,
    alpha: T,
    a: UnsafePointer<T>,
    lda: BLAS.Index,
    b: UnsafeMutablePointer<T>,
    ldb: BLAS.Index
) {
    let rows = _blasCount(m)
    let columns = _blasCount(n)
    if alpha == .zero {
        for row in 0..<rows {
            for column in 0..<columns {
                b[_matrixIndex(layout, row, column, ldb)] = .zero
            }
        }
        return
    }
    var source = Array(repeating: T.zero, count: rows * columns)
    for row in 0..<rows {
        for column in 0..<columns {
            source[row * columns + column] = b[_matrixIndex(layout, row, column, ldb)]
        }
    }
    let inner = side == .left ? rows : columns
    for row in 0..<rows {
        for column in 0..<columns {
            var sum = T.zero
            for p in 0..<inner {
                if side == .left {
                    sum += _opTriangularElement(transpose: transpose, row: row, column: p) {
                        _triangularElement(a, layout: layout, triangle: triangle, diagonal: diagonal, row: $0, column: $1, leadingDimension: lda)
                    } * source[p * columns + column]
                } else {
                    sum += source[row * columns + p] * _opTriangularElement(transpose: transpose, row: p, column: column) {
                        _triangularElement(a, layout: layout, triangle: triangle, diagonal: diagonal, row: $0, column: $1, leadingDimension: lda)
                    }
                }
            }
            b[_matrixIndex(layout, row, column, ldb)] = alpha * sum
        }
    }
}

@inlinable
internal func _trsm<T: BLASScalar>(
    layout: BLAS.Layout,
    side: BLAS.Side,
    triangle: BLAS.Triangle,
    transpose: BLAS.Transpose,
    diagonal: BLAS.Diagonal,
    m: BLAS.Index,
    n: BLAS.Index,
    alpha: T,
    a: UnsafePointer<T>,
    lda: BLAS.Index,
    b: UnsafeMutablePointer<T>,
    ldb: BLAS.Index
) {
    let rows = _blasCount(m)
    let columns = _blasCount(n)
    guard rows > 0 && columns > 0 else { return }
    if alpha == .zero {
        for row in 0..<rows {
            for column in 0..<columns {
                b[_matrixIndex(layout, row, column, ldb)] = .zero
            }
        }
        return
    }
    for row in 0..<rows {
        for column in 0..<columns {
            let index = _matrixIndex(layout, row, column, ldb)
            b[index] = alpha * b[index]
        }
    }

    let opElement: (Int, Int) -> T = { row, column in
        _opTriangularElement(transpose: transpose, row: row, column: column) {
            _triangularElement(a, layout: layout, triangle: triangle, diagonal: diagonal, row: $0, column: $1, leadingDimension: lda)
        }
    }
    let effectiveUpper = _isTransposed(transpose) ? triangle == .lower : triangle == .upper

    if side == .left {
        for column in 0..<columns {
            if effectiveUpper {
                for row in stride(from: rows - 1, through: 0, by: -1) {
                    var value = b[_matrixIndex(layout, row, column, ldb)]
                    if row + 1 < rows {
                        for p in (row + 1)..<rows {
                            value -= opElement(row, p) * b[_matrixIndex(layout, p, column, ldb)]
                        }
                    }
                    b[_matrixIndex(layout, row, column, ldb)] = value / opElement(row, row)
                }
            } else {
                for row in 0..<rows {
                    var value = b[_matrixIndex(layout, row, column, ldb)]
                    if row > 0 {
                        for p in 0..<row {
                            value -= opElement(row, p) * b[_matrixIndex(layout, p, column, ldb)]
                        }
                    }
                    b[_matrixIndex(layout, row, column, ldb)] = value / opElement(row, row)
                }
            }
        }
    } else {
        for row in 0..<rows {
            if effectiveUpper {
                for column in 0..<columns {
                    var value = b[_matrixIndex(layout, row, column, ldb)]
                    if column > 0 {
                        for p in 0..<column {
                            value -= b[_matrixIndex(layout, row, p, ldb)] * opElement(p, column)
                        }
                    }
                    b[_matrixIndex(layout, row, column, ldb)] = value / opElement(column, column)
                }
            } else {
                for column in stride(from: columns - 1, through: 0, by: -1) {
                    var value = b[_matrixIndex(layout, row, column, ldb)]
                    if column + 1 < columns {
                        for p in (column + 1)..<columns {
                            value -= b[_matrixIndex(layout, row, p, ldb)] * opElement(p, column)
                        }
                    }
                    b[_matrixIndex(layout, row, column, ldb)] = value / opElement(column, column)
                }
            }
        }
    }
}
