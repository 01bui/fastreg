#pragma once

/* cuda includes */
#include <cuda_runtime_api.h>

/* sys headers */
#include <cstddef>

namespace fastreg {
namespace cuda {

template <typename T>
struct DevPtr {
    typedef T elem_type;
    const static size_t elem_size = sizeof(elem_type);

    T *data;

    __host__ __device__ __forceinline__ DevPtr() : data(0) {}
    __host__ __device__ __forceinline__ DevPtr(T *data_arg) : data(data_arg) {}

    __host__ __device__ __forceinline__ size_t elemSize() const { return elem_size; }
    __host__ __device__ __forceinline__ operator T *() { return data; }
    __host__ __device__ __forceinline__ operator const T *() const { return data; }
};

template <typename T>
struct PtrSz : public DevPtr<T> {
    __host__ __device__ __forceinline__ PtrSz() : size(0) {}
    __host__ __device__ __forceinline__ PtrSz(T *data_arg, size_t size_arg) : DevPtr<T>(data_arg), size(size_arg) {}

    size_t size;
};

template <typename T>
struct PtrStep : public DevPtr<T> {
    __host__ __device__ __forceinline__ PtrStep() : step(0) {}
    __host__ __device__ __forceinline__ PtrStep(T *data_arg, size_t step_arg) : DevPtr<T>(data_arg), step(step_arg) {}

    /* stride between two consecutive rows in bytes, step is stored always
     * and everywhere in bytes */
    size_t step;

    __host__ __device__ __forceinline__ T *ptr(int y = 0) { return (T *) ((char *) DevPtr<T>::data + y * step); }
    __host__ __device__ __forceinline__ const T *ptr(int y = 0) const {
        return (const T *) ((const char *) DevPtr<T>::data + y * step);
    }

    __host__ __device__ __forceinline__ T &operator()(int y, int x) { return ptr(y)[x]; }
    __host__ __device__ __forceinline__ const T &operator()(int y, int x) const { return ptr(y)[x]; }
};

template <typename T>
struct PtrStepSz : public PtrStep<T> {
    __host__ __device__ __forceinline__ PtrStepSz() : cols(0), rows(0) {}
    __host__ __device__ __forceinline__ PtrStepSz(int rows_arg, int cols_arg, T *data_arg, size_t step_arg)
        : PtrStep<T>(data_arg, step_arg), cols(cols_arg), rows(rows_arg) {}

    int cols;
    int rows;
};

}  // namespace cuda

namespace device {

using fastreg::cuda::PtrStep;
using fastreg::cuda::PtrStepSz;
using fastreg::cuda::PtrSz;

}  // namespace device
}  // namespace fastreg
