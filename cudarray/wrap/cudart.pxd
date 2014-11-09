cdef extern from "driver_types.h":
    enum cudaMemcpyKind:
        cudaMemcpyHostToHost
        cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost
        cudaMemcpyDeviceToDevice
        cudaMemcpyDefault

    enum cudaError:
        cudaSuccess

    ctypedef cudaError cudaError_t

    ctypedef struct CUstream_st:
        pass
    ctypedef CUstream_st *cudaStream_t


cdef extern from "cuda_runtime_api.h":
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                           cudaMemcpyKind kind)
    cudaError_t cudaMalloc(void **devPtr, size_t size)
    cudaError_t cudaFree(void *devPtr)
    const char* cudaGetErrorString(cudaError_t error)

    cudaError_t cudaDeviceSynchronize()
    cudaError_t cudaGetLastError()

    cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                cudaMemcpyKind kind, cudaStream_t stream=*)
    cudaError_t cudaSetDevice(int device) 	


cpdef initialize(int device_id)
cdef cudaCheck(cudaError_t status)
cpdef cudaSyncCheck()
