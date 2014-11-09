from cudart cimport *


cpdef initialize(int device_id):
    cudaCheck(cudaSetDevice(device_id))
    # Establish context
    cudaCheck(cudaFree(<void *>0))


cdef cudaCheck(cudaError_t status):
    if status == cudaSuccess:
        return 0
    else:
        raise ValueError(cudaGetErrorString(status))


cpdef cudaSyncCheck():
    cudaCheck(cudaDeviceSynchronize())
    cudaCheck(cudaGetLastError())    
