// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

@inlinable
internal func _omatcopy<T: BLASScalar>(
    layout: BLAS.Layout,
    transpose: BLAS.Transpose,
    rows: BLAS.Index,
    columns: BLAS.Index,
    alpha: T,
    a: UnsafePointer<T>,
    lda: BLAS.Index,
    b: UnsafeMutablePointer<T>,
    ldb: BLAS.Index
) {
    let sourceRows = _blasCount(rows)
    let sourceColumns = _blasCount(columns)
    let outputRows = _isTransposed(transpose) ? sourceColumns : sourceRows
    let outputColumns = _isTransposed(transpose) ? sourceRows : sourceColumns
    for row in 0..<outputRows {
        for column in 0..<outputColumns {
            let index = _matrixIndex(layout, row, column, ldb)
            if alpha == .zero {
                b[index] = .zero
            } else {
                b[index] = alpha * _opElement(
                    a, layout: layout, transpose: transpose,
                    row: row, column: column, leadingDimension: lda
                )
            }
        }
    }
}

@inlinable
internal func _imatcopy<T: BLASScalar>(
    layout: BLAS.Layout,
    transpose: BLAS.Transpose,
    rows: BLAS.Index,
    columns: BLAS.Index,
    alpha: T,
    a: UnsafeMutablePointer<T>,
    lda: BLAS.Index,
    ldb: BLAS.Index
) {
    let sourceRows = _blasCount(rows)
    let sourceColumns = _blasCount(columns)
    guard sourceRows > 0 && sourceColumns > 0 else { return }
    let sourceLDA = layout == .rowMajor ? sourceColumns : sourceRows
    var source = Array(repeating: T.zero, count: sourceRows * sourceColumns)
    for row in 0..<sourceRows {
        for column in 0..<sourceColumns {
            source[_matrixIndex(layout, row, column, sourceLDA)] = a[_matrixIndex(layout, row, column, lda)]
        }
    }
    source.withUnsafeBufferPointer { buffer in
        _omatcopy(
            layout: layout, transpose: transpose,
            rows: rows, columns: columns, alpha: alpha,
            a: buffer.baseAddress!, lda: sourceLDA,
            b: a, ldb: ldb
        )
    }
}

@inlinable
internal func _geadd<T: BLASScalar>(
    layout: BLAS.Layout,
    transposeA: BLAS.Transpose,
    transposeC: BLAS.Transpose,
    rows: BLAS.Index,
    columns: BLAS.Index,
    alpha: T,
    a: UnsafePointer<T>,
    lda: BLAS.Index,
    beta: T,
    c: UnsafeMutablePointer<T>,
    ldc: BLAS.Index
) {
    let outputRows = _blasCount(rows)
    let outputColumns = _blasCount(columns)
    for row in 0..<outputRows {
        for column in 0..<outputColumns {
            let source = alpha == .zero ? T.zero : _opElement(
                a, layout: layout, transpose: transposeA,
                row: row, column: column, leadingDimension: lda
            )
            let storedRow = _isTransposed(transposeC) ? column : row
            let storedColumn = _isTransposed(transposeC) ? row : column
            let index = _matrixIndex(layout, storedRow, storedColumn, ldc)
            let logicalC = _isConjugated(transposeC) ? c[index].conjugate : c[index]
            var result = alpha * source + (beta == .zero ? .zero : beta * logicalC)
            if _isConjugated(transposeC) { result = result.conjugate }
            c[index] = result
        }
    }
}
