// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule

@usableFromInline
internal protocol BLASScalar: AlgebraicField where Magnitude: Real {
    static var one: Self { get }
    
    var conjugate: Self { get }
    var abs1: Magnitude { get }
    var normSquared: Magnitude { get }
    var sumComponents: Magnitude { get }
    var comparisonValue: Magnitude { get }
    var withZeroImaginary: Self { get }
    
    func scaled(by value: Magnitude) -> Self
}

extension BLASScalar {
    @inlinable
    @inline(always)
    @_transparent
    internal static var one: Self { 1 }
}

extension Float: BLASScalar {
    @usableFromInline
    typealias RealType = Self
    
    @inlinable
    @inline(always)
    @_transparent
    internal var conjugate: Float { self }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var abs1: Float { magnitude }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var normSquared: Float { self * self }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var sumComponents: Float { self }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var comparisonValue: Float { self }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var withZeroImaginary: Float { self }
    
    @inlinable
    @inline(always)
    @_transparent
    internal func scaled(by value: Float) -> Float { self * value }
}

extension Double: BLASScalar {
    @usableFromInline
    typealias RealType = Self
    
    @inlinable
    @inline(always)
    @_transparent
    internal static var one: Double { 1 }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var conjugate: Double { self }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var abs1: Double { magnitude }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var normSquared: Double { self * self }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var sumComponents: Double { self }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var comparisonValue: Double { self }
    
    @inlinable
    @inline(always)
    @_transparent
    internal var withZeroImaginary: Double { self }
    
    @inlinable
    @inline(always)
    @_transparent
    internal func scaled(by value: Double) -> Double { self * value }
}

extension Complex: BLASScalar {
    @inlinable
    @inline(always)
    @_transparent
    var abs1: RealType {
        real.magnitude + imaginary.magnitude
    }
    
    @inlinable
    @inline(always)
    @_transparent
    var normSquared: RealType {
        real * real + imaginary * imaginary
    }
    
    @inlinable
    @inline(always)
    @_transparent
    var sumComponents: RealType {
        real + imaginary
    }
    
    @inlinable
    @inline(always)
    @_transparent
    var comparisonValue: RealType {
        real + imaginary
    }
    
    @inlinable
    @inline(always)
    @_transparent
    var withZeroImaginary: ComplexModule.Complex<RealType> {
        .init(real, .zero)
    }
    
    @inlinable
    @inline(always)
    @_transparent
    func scaled(by value: RealType) -> ComplexModule.Complex<RealType> {
        .init(real * value, imaginary * value)
    }
}

@inlinable
@inline(always)
@_transparent
internal func _complexFloatPointer(
    _ pointer: UnsafePointer<Complex<Float>>
) -> UnsafeRawPointer {
    UnsafeRawPointer(pointer)
}

@inlinable
@inline(always)
@_transparent
internal func _complexFloatPointer(
    _ pointer: UnsafePointer<Complex<Float>>
) -> OpaquePointer {
    OpaquePointer(pointer)
}

@inlinable
@inline(always)
@_transparent
internal func _complexFloatPointer(
    _ pointer: UnsafeMutablePointer<Complex<Float>>
) -> UnsafeMutableRawPointer {
    UnsafeMutableRawPointer(pointer)
}

@inlinable
@inline(always)
@_transparent
internal func _complexFloatPointer(
    _ pointer: UnsafeMutablePointer<Complex<Float>>
) -> OpaquePointer {
    OpaquePointer(pointer)
}

@inlinable
@inline(always)
@_transparent
internal func _complexDoublePointer(
    _ pointer: UnsafePointer<Complex<Double>>
) -> UnsafeRawPointer {
    UnsafeRawPointer(pointer)
}

@inlinable
@inline(always)
@_transparent
internal func _complexDoublePointer(
    _ pointer: UnsafePointer<Complex<Double>>
) -> OpaquePointer {
    OpaquePointer(pointer)
}

@inlinable
@inline(always)
@_transparent
internal func _complexDoublePointer(
    _ pointer: UnsafeMutablePointer<Complex<Double>>
) -> UnsafeMutableRawPointer {
    UnsafeMutableRawPointer(pointer)
}

@inlinable
@inline(always)
@_transparent
internal func _complexDoublePointer(
    _ pointer: UnsafeMutablePointer<Complex<Double>>
) -> OpaquePointer {
    OpaquePointer(pointer)
}
