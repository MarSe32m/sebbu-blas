// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

@inlinable
@inline(always)
@_transparent
internal func _blasCount(_ value: BLAS.Index) -> Int {
    value > 0 ? value : 0
}

@inlinable
@inline(always)
@_transparent
internal func _vectorStart(_ count: Int, _ stride: BLAS.Index) -> Int {
    stride < 0 ? (1 - count) * stride : 0
}

@inlinable
internal func _matrixIndex(
    _ layout: BLAS.Layout,
    _ row: Int,
    _ column: Int,
    _ leadingDimension: BLAS.Index
) -> Int {
    switch layout {
    case .rowMajor: row * leadingDimension + column
    case .columnMajor: column * leadingDimension + row
    }
}

@inlinable
@inline(always)
@_transparent
internal func _isTransposed(_ transpose: BLAS.Transpose) -> Bool {
    transpose == .transpose || transpose == .conjugateTranspose
}

@inlinable
@inline(always)
@_transparent
internal func _isConjugated(_ transpose: BLAS.Transpose) -> Bool {
    transpose == .conjugateNoTranspose || transpose == .conjugateTranspose
}

@inlinable
internal func _opElement<T: BLASScalar>(
    _ pointer: UnsafePointer<T>,
    layout: BLAS.Layout,
    transpose: BLAS.Transpose,
    row: Int,
    column: Int,
    leadingDimension: BLAS.Index
) -> T {
    let sourceRow = _isTransposed(transpose) ? column : row
    let sourceColumn = _isTransposed(transpose) ? row : column
    let value = pointer[_matrixIndex(layout, sourceRow, sourceColumn, leadingDimension)]
    return _isConjugated(transpose) ? value.conjugate : value
}

@inlinable
internal func _symmetricElement<T: BLASScalar>(
    _ pointer: UnsafePointer<T>,
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    row: Int,
    column: Int,
    leadingDimension: BLAS.Index,
    hermitian: Bool
) -> T {
    let stored = triangle == .upper ? row <= column : row >= column
    let sourceRow = stored ? row : column
    let sourceColumn = stored ? column : row
    let value = pointer[_matrixIndex(layout, sourceRow, sourceColumn, leadingDimension)]
    if hermitian && row == column { return value.withZeroImaginary }
    return hermitian && !stored ? value.conjugate : value
}

@inlinable
internal func _packedIndex(
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    order: Int,
    row: Int,
    column: Int
) -> Int {
    switch (layout, triangle) {
    case (.columnMajor, .upper):
        return column * (column + 1) / 2 + row
    case (.columnMajor, .lower):
        return column * order - column * (column - 1) / 2 + row - column
    case (.rowMajor, .upper):
        return row * order - row * (row - 1) / 2 + column - row
    case (.rowMajor, .lower):
        return row * (row + 1) / 2 + column
    }
}

@inlinable
internal func _packedElement<T: BLASScalar>(
    _ pointer: UnsafePointer<T>,
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    order: Int,
    row: Int,
    column: Int,
    hermitian: Bool
) -> T {
    let stored = triangle == .upper ? row <= column : row >= column
    let sourceRow = stored ? row : column
    let sourceColumn = stored ? column : row
    let index = _packedIndex(
        layout: layout,
        triangle: triangle,
        order: order,
        row: sourceRow,
        column: sourceColumn
    )
    let value = pointer[index]
    if hermitian && row == column { return value.withZeroImaginary }
    return hermitian && !stored ? value.conjugate : value
}

@inlinable
internal func _generalBandElement<T: BLASScalar>(
    _ pointer: UnsafePointer<T>,
    layout: BLAS.Layout,
    rows: Int,
    columns: Int,
    lowerBandwidth: Int,
    upperBandwidth: Int,
    row: Int,
    column: Int,
    leadingDimension: BLAS.Index
) -> T {
    guard row >= 0, row < rows, column >= 0, column < columns,
          row - column <= lowerBandwidth, column - row <= upperBandwidth else {
        return .zero
    }
    switch layout {
    case .columnMajor:
        return pointer[column * leadingDimension + upperBandwidth + row - column]
    case .rowMajor:
        return pointer[row * leadingDimension + lowerBandwidth + column - row]
    }
}

@inlinable
internal func _symmetricBandElement<T: BLASScalar>(
    _ pointer: UnsafePointer<T>,
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    order: Int,
    bandwidth: Int,
    row: Int,
    column: Int,
    leadingDimension: BLAS.Index,
    hermitian: Bool
) -> T {
    guard row >= 0, row < order, column >= 0, column < order,
          Swift.abs(row - column) <= bandwidth else { return .zero }

    let stored = triangle == .upper ? row <= column : row >= column
    let sourceRow = stored ? row : column
    let sourceColumn = stored ? column : row
    let index: Int
    switch (layout, triangle) {
    case (.columnMajor, .upper):
        index = sourceColumn * leadingDimension + bandwidth + sourceRow - sourceColumn
    case (.columnMajor, .lower):
        index = sourceColumn * leadingDimension + sourceRow - sourceColumn
    case (.rowMajor, .upper):
        index = sourceRow * leadingDimension + sourceColumn - sourceRow
    case (.rowMajor, .lower):
        index = sourceRow * leadingDimension + bandwidth + sourceColumn - sourceRow
    }
    let value = pointer[index]
    if hermitian && row == column { return value.withZeroImaginary }
    return hermitian && !stored ? value.conjugate : value
}

@inlinable
internal func _triangularElement<T: BLASScalar>(
    _ pointer: UnsafePointer<T>,
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    diagonal: BLAS.Diagonal,
    row: Int,
    column: Int,
    leadingDimension: BLAS.Index
) -> T {
    if row == column && diagonal == .unit { return T.one }
    let present = triangle == .upper ? row <= column : row >= column
    return present ? pointer[_matrixIndex(layout, row, column, leadingDimension)] : .zero
}

@inlinable
internal func _triangularBandElement<T: BLASScalar>(
    _ pointer: UnsafePointer<T>,
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    diagonal: BLAS.Diagonal,
    order: Int,
    bandwidth: Int,
    row: Int,
    column: Int,
    leadingDimension: BLAS.Index
) -> T {
    if row == column && diagonal == .unit { return T.one }
    let present = triangle == .upper ? row <= column : row >= column
    guard present, row >= 0, row < order, column >= 0, column < order,
          Swift.abs(row - column) <= bandwidth else { return .zero }

    let index: Int
    switch (layout, triangle) {
    case (.columnMajor, .upper):
        index = column * leadingDimension + bandwidth + row - column
    case (.columnMajor, .lower):
        index = column * leadingDimension + row - column
    case (.rowMajor, .upper):
        index = row * leadingDimension + column - row
    case (.rowMajor, .lower):
        index = row * leadingDimension + bandwidth + column - row
    }
    return pointer[index]
}

@inlinable
internal func _triangularPackedElement<T: BLASScalar>(
    _ pointer: UnsafePointer<T>,
    layout: BLAS.Layout,
    triangle: BLAS.Triangle,
    diagonal: BLAS.Diagonal,
    order: Int,
    row: Int,
    column: Int
) -> T {
    if row == column && diagonal == .unit { return T.one }
    let present = triangle == .upper ? row <= column : row >= column
    guard present else { return .zero }
    return pointer[_packedIndex(
        layout: layout,
        triangle: triangle,
        order: order,
        row: row,
        column: column
    )]
}
