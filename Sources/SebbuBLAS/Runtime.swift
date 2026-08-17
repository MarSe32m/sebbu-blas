// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

#if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
import COpenBLAS
#endif

import Synchronization

extension BLAS {
    /// The parallelization model compiled into OpenBLAS.
    @frozen
    public enum Parallelization: Int, Sendable {
        case sequential = 0
        case thread = 1
        case openMP = 2
    }

    /// A temporary Swift view of OpenBLAS's `dojob` function pointer.
    ///
    /// This value and every job-data pointer supplied with it are valid only
    /// for the duration of the enclosing ``ThreadsCallback`` invocation.
    public struct OpenBLASJob: @unchecked Sendable {
        @usableFromInline
        internal typealias Invocation = (
            _ threadNumber: Int,
            _ jobData: UnsafeMutableRawPointer?,
            _ data: Int
        ) -> Void

        @usableFromInline
        internal let invocation: Invocation

        @usableFromInline
        internal init(_ invocation: @escaping Invocation) {
            self.invocation = invocation
        }

        /// Runs one OpenBLAS job.
        @inlinable
        public func callAsFunction(
            threadNumber: Int,
            jobData: UnsafeMutableRawPointer?,
            data: Int
        ) {
            invocation(threadNumber, jobData, data)
        }
    }

    /// A replacement for OpenBLAS's internal thread scheduler.
    ///
    /// OpenBLAS requires this callback to be installed before any other
    /// OpenBLAS routine is called. The callback and all values passed to it
    /// must not escape the invocation.
    public typealias ThreadsCallback = @Sendable (
        _ synchronous: Int,
        _ doJob: OpenBLASJob,
        _ numberOfJobs: Int,
        _ jobDataElementSize: Int,
        _ jobData: UnsafeMutableRawPointer?,
        _ data: Int
    ) -> Void

    /// A thread-safe handler for an invalid BLAS parameter.
    ///
    /// The routine-name buffer is not NUL-terminated and is valid only while
    /// the handler is running. OpenBLAS may invoke the handler concurrently.
    public typealias XerblaHandler = @Sendable (
        _ routineName: UnsafeBufferPointer<CChar>,
        _ parameter: Int
    ) -> Void
}

@usableFromInline
internal final class _OpenBLASCallbackBox<Callback>: @unchecked Sendable {
    @usableFromInline
    internal let callback: Callback

    @usableFromInline
    internal init(_ callback: Callback) {
        self.callback = callback
    }
}

private let _openBLASThreadsCallbackStorage =
    Mutex<_OpenBLASCallbackBox<BLAS.ThreadsCallback>?>(nil)
private let _openBLASXerblaHandlerStorage =
    Mutex<_OpenBLASCallbackBox<BLAS.XerblaHandler>?>(nil)

#if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
private let _openBLASThreadsCallbackBridge: openblas_threads_callback = {
    synchronous,
    doJob,
    numberOfJobs,
    jobDataElementSize,
    jobData,
    data in

    guard let doJob else { return }
    guard let callback = _openBLASThreadsCallbackStorage.withLock({ $0 }) else {
        return
    }

    let swiftJob = BLAS.OpenBLASJob { threadNumber, jobData, data in
        doJob(_backendIndex(threadNumber), jobData, _backendIndex(data))
    }
    callback.callback(
        Int(synchronous),
        swiftJob,
        Int(numberOfJobs),
        Int(jobDataElementSize),
        jobData,
        Int(data)
    )
}

private let _openBLASXerblaHandlerBridge: openblas_xerbla_handler = {
    routineName,
    parameter,
    nameLength in

    guard let routineName, let parameter else { return }
    _invokeXerblaHandler(
        routineName: UnsafeBufferPointer(
            start: routineName,
            count: Int(nameLength)
        ),
        parameter: Int(parameter.pointee)
    )
}
#endif

@inline(never)
internal func _invokeXerblaHandler(
    routineName: UnsafeBufferPointer<CChar>,
    parameter: Int
) {
    let handler = _openBLASXerblaHandlerStorage.withLock { $0 }
    handler?.callback(routineName, parameter)
}

extension BLAS {
    /// Sets OpenBLAS's process-wide thread count.
    @inlinable
    public static func setNumThreads(_ numThreads: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        openblas_set_num_threads(_backendIndex(numThreads))
        #else
        _ = numThreads
        #endif
    }

    /// Compatibility spelling for OpenBLAS's legacy thread-count setter.
    @inlinable
    public static func gotoSetNumThreads(_ numThreads: Int) {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        goto_set_num_threads(_backendIndex(numThreads))
        #else
        _ = numThreads
        #endif
    }

    /// Sets the calling context's local thread count and returns its previous value.
    @inlinable
    @discardableResult
    public static func setNumThreadsLocal(_ numThreads: Int) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(openblas_set_num_threads_local(_backendIndex(numThreads)))
        #else
        _ = numThreads
        return 1
        #endif
    }

    /// Returns the active backend's thread count.
    @inlinable
    public static func getNumThreads() -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(openblas_get_num_threads())
        #else
        1
        #endif
    }

    /// Returns the processor count reported by OpenBLAS, or one for a portable backend.
    @inlinable
    public static func getNumProcs() -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(openblas_get_num_procs())
        #else
        1
        #endif
    }

    /// Returns the backend's build configuration string.
    public static func getConfig() -> String {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        guard let pointer = openblas_get_config() else { return "" }
        return String(cString: pointer)
        #elseif (canImport(Accelerate) && os(macOS)) && !SEBBU_BLAS_FORCE_SWIFT
        return "Accelerate"
        #else
        return "Swift fallback"
        #endif
    }

    /// Returns OpenBLAS's CPU core name, or a portable backend identifier.
    public static func getCorename() -> String {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        guard let pointer = openblas_get_corename() else { return "" }
        return String(cString: pointer)
        #elseif (canImport(Accelerate) && os(macOS)) && !SEBBU_BLAS_FORCE_SWIFT
        return "Apple"
        #else
        return "generic"
        #endif
    }

    /// Installs a custom OpenBLAS thread scheduler.
    ///
    /// The portable and Accelerate backends retain the callback for API
    /// compatibility but do not produce OpenBLAS jobs.
    public static func setThreadsCallbackFunction(_ callback: ThreadsCallback?) {
        let box: _OpenBLASCallbackBox<ThreadsCallback>? = callback.map {
            _OpenBLASCallbackBox($0)
        }
        _openBLASThreadsCallbackStorage.withLock { current in
            current = box
            #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
            if callback == nil {
                openblas_set_threads_callback_function(nil)
            } else {
                openblas_set_threads_callback_function(_openBLASThreadsCallbackBridge)
            }
            #endif
        }
    }

    /// Replaces the XERBLA handler and returns the previous Swift handler.
    ///
    /// Passing `nil` restores OpenBLAS's default handler. A returned handler
    /// retains its captures so callers can keep it alive while any in-flight
    /// invocation finishes.
    @discardableResult
    public static func setXerbla(_ handler: XerblaHandler?) -> XerblaHandler? {
        let box: _OpenBLASCallbackBox<XerblaHandler>? = handler.map {
            _OpenBLASCallbackBox($0)
        }
        let previous = _openBLASXerblaHandlerStorage.withLock { current in
            let previous = current
            #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
            if handler == nil {
                _ = openblas_set_xerbla(nil)
            } else {
                _ = openblas_set_xerbla(_openBLASXerblaHandlerBridge)
            }
            #endif
            current = box
            return previous
        }
        return previous?.callback
    }

    /// Returns the OpenBLAS parallelization constant as a Swift `Int`.
    @inlinable
    public static func getParallel() -> Int {
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        Int(openblas_get_parallel())
        #else
        Parallelization.sequential.rawValue
        #endif
    }
}

#if os(Linux)
extension BLAS {
    /// Sets affinity for one OpenBLAS thread.
    ///
    /// `cpuSet` must point to an aligned Linux `cpu_set_t` allocation whose
    /// accessible size is at least `cpuSetSize` bytes.
    @inlinable
    public static func setAffinity(
        threadIndex: Int,
        cpuSetSize: Int,
        cpuSet: UnsafeMutableRawPointer
    ) -> Int {
        precondition(cpuSetSize >= 0, "A CPU-set size cannot be negative")
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        return Int(
            openblas_setaffinity(
                _backendIndex(threadIndex),
                cpuSetSize,
                cpuSet.assumingMemoryBound(to: cpu_set_t.self)
            )
        )
        #else
        _ = threadIndex
        _ = cpuSet
        return -1
        #endif
    }

    /// Queries affinity for one OpenBLAS thread.
    ///
    /// `cpuSet` must point to an aligned Linux `cpu_set_t` allocation whose
    /// accessible size is at least `cpuSetSize` bytes.
    @inlinable
    public static func getAffinity(
        threadIndex: Int,
        cpuSetSize: Int,
        cpuSet: UnsafeMutableRawPointer
    ) -> Int {
        precondition(cpuSetSize >= 0, "A CPU-set size cannot be negative")
        #if canImport(COpenBLAS) && !SEBBU_BLAS_FORCE_SWIFT
        return Int(
            openblas_getaffinity(
                _backendIndex(threadIndex),
                cpuSetSize,
                cpuSet.assumingMemoryBound(to: cpu_set_t.self)
            )
        )
        #else
        _ = threadIndex
        _ = cpuSet
        return -1
        #endif
    }
}
#endif
