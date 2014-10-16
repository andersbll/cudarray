from cudart cimport *

cdef cudaCheck(cudaError_t status):
    if status == cudaSuccess:
        return 0
    else:
        raise ValueError(cudaGetErrorString(status))

cpdef cudaSyncCheck():
    cudaCheck(cudaDeviceSynchronize())
    cudaCheck(cudaGetLastError())    
