// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule

@inlinable
internal func _dot<T: BLASScalar>(
    _ n: BLAS.Index,
    _ x: UnsafePointer<T>,
    _ incX: BLAS.Index,
    _ y: UnsafePointer<T>,
    _ incY: BLAS.Index,
    conjugateX: Bool = false
) -> T {
    let count = _blasCount(n)
    guard count > 0 else { return .zero }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var iy = _vectorStart(count, incY)
    var result = T.zero
    for _ in 0..<count {
        let lhs = conjugateX ? x[ix].conjugate : x[ix]
        result += lhs * y[iy]
        ix += incX
        iy += incY
    }
    return result
}

@inlinable
internal func _mixedDot(
    _ n: BLAS.Index,
    _ x: UnsafePointer<Float>,
    _ incX: BLAS.Index,
    _ y: UnsafePointer<Float>,
    _ incY: BLAS.Index
) -> Double {
    let count = _blasCount(n)
    guard count > 0 else { return 0 }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var iy = _vectorStart(count, incY)
    var result = 0.0
    for _ in 0..<count {
        result += Double(x[ix]) * Double(y[iy])
        ix += incX
        iy += incY
    }
    return result
}

@inlinable
internal func _asum<T: BLASScalar>(
    _ n: BLAS.Index,
    _ x: UnsafePointer<T>,
    _ incX: BLAS.Index
) -> T.Magnitude {
    let count = _blasCount(n)
    guard count > 0 else { return 0 }
    precondition(incX != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var result: T.Magnitude = .zero
    for _ in 0..<count {
        result += x[ix].abs1
        ix += incX
    }
    return result
}

@inlinable
internal func _sum<T: BLASScalar>(
    _ n: BLAS.Index,
    _ x: UnsafePointer<T>,
    _ incX: BLAS.Index
) -> T.Magnitude {
    let count = _blasCount(n)
    guard count > 0 else { return 0 }
    precondition(incX != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var result: T.Magnitude = .zero
    for _ in 0..<count {
        result += x[ix].sumComponents
        ix += incX
    }
    return result
}

@inlinable
internal func _nrm2<T: BLASScalar>(
    _ n: BLAS.Index,
    _ x: UnsafePointer<T>,
    _ incX: BLAS.Index
) -> T.Magnitude {
    let count = _blasCount(n)
    guard count > 0 else { return 0 }
    precondition(incX != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var sum: T.Magnitude = .zero
    for _ in 0..<count {
        sum += x[ix].normSquared
        ix += incX
    }
    return sum.squareRoot()
}

@inlinable
internal func _extremeIndex<T: BLASScalar>(
    _ n: BLAS.Index,
    _ x: UnsafePointer<T>,
    _ incX: BLAS.Index,
    absolute: Bool,
    minimum: Bool
) -> Int {
    let count = _blasCount(n)
    guard count > 0, incX != 0 else { return 0 }
    var ix = _vectorStart(count, incX)
    var bestIndex = 0
    var best = absolute ? x[ix].abs1 : x[ix].comparisonValue
    for logicalIndex in 1..<count {
        ix += incX
        let value = absolute ? x[ix].abs1 : x[ix].comparisonValue
        if minimum ? value < best : value > best {
            best = value
            bestIndex = logicalIndex
        }
    }
    return bestIndex
}

@inlinable
internal func _extremeValue<T: BLASScalar>(
    _ n: BLAS.Index,
    _ x: UnsafePointer<T>,
    _ incX: BLAS.Index,
    absolute: Bool,
    minimum: Bool
) -> T.Magnitude {
    let count = _blasCount(n)
    guard count > 0, incX != 0 else { return 0 }
    var ix = _vectorStart(count, incX)
    var best = absolute ? x[ix].abs1 : x[ix].comparisonValue
    for _ in 1..<count {
        ix += incX
        let value = absolute ? x[ix].abs1 : x[ix].comparisonValue
        if minimum ? value < best : value > best { best = value }
    }
    return best
}

@inlinable
internal func _axpy<T: BLASScalar>(
    _ n: BLAS.Index,
    alpha: T,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index,
    conjugateX: Bool = false
) {
    let count = _blasCount(n)
    guard count > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var iy = _vectorStart(count, incY)
    for _ in 0..<count {
        let value = conjugateX ? x[ix].conjugate : x[ix]
        y[iy] += alpha * value
        ix += incX
        iy += incY
    }
}

@inlinable
internal func _axpby<T: BLASScalar>(
    _ n: BLAS.Index,
    alpha: T,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    beta: T,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index
) {
    let count = _blasCount(n)
    guard count > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var iy = _vectorStart(count, incY)
    for _ in 0..<count {
        y[iy] = alpha * x[ix] + beta * y[iy]
        ix += incX
        iy += incY
    }
}

@inlinable
internal func _copy<T>(
    _ n: BLAS.Index,
    x: UnsafePointer<T>,
    incX: BLAS.Index,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index
) {
    let count = _blasCount(n)
    guard count > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var iy = _vectorStart(count, incY)
    for _ in 0..<count {
        y[iy] = x[ix]
        ix += incX
        iy += incY
    }
}

@inlinable
internal func _swap<T>(
    _ n: BLAS.Index,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index
) {
    let count = _blasCount(n)
    guard count > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var iy = _vectorStart(count, incY)
    for _ in 0..<count {
        let temporary = x[ix]
        x[ix] = y[iy]
        y[iy] = temporary
        ix += incX
        iy += incY
    }
}

@inlinable
internal func _rot<T: BLASScalar>(
    _ n: BLAS.Index,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index,
    c: T.Magnitude,
    s: T.Magnitude
) {
    let count = _blasCount(n)
    guard count > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var iy = _vectorStart(count, incY)
    for _ in 0..<count {
        let oldX = x[ix]
        let oldY = y[iy]
        x[ix] = oldX.scaled(by: c) + oldY.scaled(by: s)
        y[iy] = oldY.scaled(by: c) - oldX.scaled(by: s)
        ix += incX
        iy += incY
    }
}

@inlinable
internal func _scal<T: BLASScalar>(
    _ n: BLAS.Index,
    alpha: T,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index
) {
    let count = _blasCount(n)
    guard count > 0 else { return }
    precondition(incX != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    for _ in 0..<count {
        x[ix] = alpha * x[ix]
        ix += incX
    }
}

@inlinable
internal func _realScal<T: BLASScalar>(
    _ n: BLAS.Index,
    alpha: T.Magnitude,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index
) {
    let count = _blasCount(n)
    guard count > 0 else { return }
    precondition(incX != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    for _ in 0..<count {
        x[ix] = x[ix].scaled(by: alpha)
        ix += incX
    }
}

@inlinable
internal func _rotg<T: Real>(
    a: UnsafeMutablePointer<T>,
    b: UnsafeMutablePointer<T>,
    c: UnsafeMutablePointer<T>,
    s: UnsafeMutablePointer<T>
) {
    let av = a.pointee
    let bv = b.pointee
    let scale = av.magnitude + bv.magnitude
    if scale == 0 {
        c.pointee = 1
        s.pointee = 0
        a.pointee = 0
        b.pointee = 0
        return
    }
    let roe = av.magnitude > bv.magnitude ? av : bv
    let scaledA = av / scale
    let scaledB = bv / scale
    var r = scale * (scaledA * scaledA + scaledB * scaledB).squareRoot()
    if roe < 0 { r = -r }
    c.pointee = av / r
    s.pointee = bv / r
    var z: T = 1
    if av.magnitude > bv.magnitude { z = s.pointee }
    if bv.magnitude >= av.magnitude && c.pointee != 0 { z = 1 / c.pointee }
    a.pointee = r
    b.pointee = z
}

@inlinable
internal func _complexRotg<T: BLASScalar>(
    a: UnsafeMutablePointer<T>,
    b: UnsafePointer<T>,
    c: UnsafeMutablePointer<T.Magnitude>,
    s: UnsafeMutablePointer<T>
) {
    let av = a.pointee
    let bv = b.pointee
    let absA = av.normSquared.squareRoot()
    let absB = bv.normSquared.squareRoot()
    if absA == 0 {
        c.pointee = 0
        s.pointee = T.one
        a.pointee = bv
        return
    }
    let scale = absA + absB
    let norm = scale * ((absA / scale) * (absA / scale) + (absB / scale) * (absB / scale)).squareRoot()
    let alpha = av.scaled(by: 1 / absA)
    c.pointee = absA / norm
    s.pointee = (alpha * bv.conjugate).scaled(by: 1 / norm)
    a.pointee = alpha.scaled(by: norm)
}

@inlinable
internal func _rotm<T: Real>(
    _ n: BLAS.Index,
    x: UnsafeMutablePointer<T>,
    incX: BLAS.Index,
    y: UnsafeMutablePointer<T>,
    incY: BLAS.Index,
    parameters p: UnsafePointer<T>
) {
    let flag = p[0]
    if flag == -2 { return }
    let h11: T
    let h21: T
    let h12: T
    let h22: T
    if flag < 0 {
        h11 = p[1]; h21 = p[2]; h12 = p[3]; h22 = p[4]
    } else if flag == 0 {
        h11 = 1; h21 = p[2]; h12 = p[3]; h22 = 1
    } else {
        h11 = p[1]; h21 = -1; h12 = 1; h22 = p[4]
    }
    let count = _blasCount(n)
    guard count > 0 else { return }
    precondition(incX != 0 && incY != 0, "BLAS vector strides must be nonzero")
    var ix = _vectorStart(count, incX)
    var iy = _vectorStart(count, incY)
    for _ in 0..<count {
        let w = x[ix]
        let z = y[iy]
        x[ix] = w * h11 + z * h12
        y[iy] = w * h21 + z * h22
        ix += incX
        iy += incY
    }
}

@inlinable
internal func _rotmg<T: Real>(
    d1: UnsafeMutablePointer<T>,
    d2: UnsafeMutablePointer<T>,
    b1: UnsafeMutablePointer<T>,
    b2: T,
    parameters p: UnsafeMutablePointer<T>
) {
    let gamma: T = 4096
    let gammaSquared = gamma * gamma
    let reciprocalGammaSquared = 1 / gammaSquared
    var flag: T = -1
    var h11: T = 0
    var h12: T = 0
    var h21: T = 0
    var h22: T = 0
    var d1v = d1.pointee
    var d2v = d2.pointee
    var x1 = b1.pointee

    if d1v < 0 {
        d1v = 0; d2v = 0; x1 = 0
    } else {
        let p2 = d2v * b2
        if p2 == 0 {
            p[0] = -2
            return
        }
        let p1 = d1v * x1
        let q2 = p2 * b2
        let q1 = p1 * x1
        if q1.magnitude > q2.magnitude {
            h21 = -b2 / x1
            h12 = p2 / p1
            let u = 1 - h12 * h21
            if u > 0 {
                flag = 0
                d1v /= u
                d2v /= u
                x1 *= u
            }
        } else if q2 < 0 {
            d1v = 0; d2v = 0; x1 = 0
        } else {
            flag = 1
            h11 = p1 / p2
            h22 = x1 / b2
            let u = 1 + h11 * h22
            let temporary = d2v / u
            d2v = d1v / u
            d1v = temporary
            x1 = b2 * u
        }

        if d1v != 0 {
            while d1v <= reciprocalGammaSquared || d1v >= gammaSquared {
                if flag == 0 { h11 = 1; h22 = 1; flag = -1 }
                else if flag == 1 { h21 = -1; h12 = 1; flag = -1 }
                if d1v <= reciprocalGammaSquared {
                    d1v *= gammaSquared
                    x1 /= gamma
                    h11 /= gamma
                    h12 /= gamma
                } else {
                    d1v /= gammaSquared
                    x1 *= gamma
                    h11 *= gamma
                    h12 *= gamma
                }
            }
        }
        if d2v != 0 {
            while d2v.magnitude <= reciprocalGammaSquared || d2v.magnitude >= gammaSquared {
                if flag == 0 { h11 = 1; h22 = 1; flag = -1 }
                else if flag == 1 { h21 = -1; h12 = 1; flag = -1 }
                if d2v.magnitude <= reciprocalGammaSquared {
                    d2v *= gammaSquared
                    h21 /= gamma
                    h22 /= gamma
                } else {
                    d2v /= gammaSquared
                    h21 *= gamma
                    h22 *= gamma
                }
            }
        }
    }

    d1.pointee = d1v
    d2.pointee = d2v
    b1.pointee = x1
    p[0] = flag
    if flag < 0 { p[1] = h11; p[2] = h21; p[3] = h12; p[4] = h22 }
    else if flag == 0 { p[2] = h21; p[3] = h12 }
    else { p[1] = h11; p[4] = h22 }
}
