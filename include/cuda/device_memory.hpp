#pragma once

/* fastreg includes */
#include <cuda/kernel_containers.hpp>

namespace fastreg {
namespace cuda {

/* error handler; all GPU functions from this subsystem call the
 * function to report an error; for internal use only */
void error(const char *error_string, const char *file, const int line, const char *func = "");

/* DeviceMemory class */
class DeviceMemory {
public:
    /* empty constructor */
    DeviceMemory();

    /* destructor */
    ~DeviceMemory();

    /* allocates internal buffer in GPU memory */
    DeviceMemory(size_t sizeBytes_arg);

    /* initializes with user allocated buffer; reference counting is
     * disabled in this case */
    DeviceMemory(void *ptr_arg, size_t sizeBytes_arg);

    /* copy constructor; just increments reference counter */
    DeviceMemory(const DeviceMemory &other_arg);

    /* assigment operator; just increments reference counter */
    DeviceMemory &operator=(const DeviceMemory &other_arg);

    /* allocates internal buffer in GPU memory; if internal buffer was created before hte function recreates it with new
     * size; if new and old sizes are equal it does nothing */
    void create(size_t sizeBytes_arg);

    /* decremenets reference counter and releases internal buffer if needed */
    void release();

    /* performs data copying; if destination size differs it will be
     * reallocated */
    void copyTo(DeviceMemory &other) const;

    /* uploads data to internal buffer in GPU memory; it calls create()
     * inside to ensure that intenal buffer size is enough */
    void upload(const void *host_ptr_arg, size_t sizeBytes_arg);

    /* downloads data from internal buffer to CPU memory */
    void download(void *host_ptr_arg) const;

    /* performs swap of data pointed with another device memory */
    void swap(DeviceMemory &other_arg);

    /* returns pointer for internal buffer in GPU memory */
    template <class T>
    T *ptr();

    /* returns constant pointer for internal buffer in GPU memory */
    template <class T>
    const T *ptr() const;

    /* conversion to PtrSz for passing to kernel functions */
    template <class U>
    operator PtrSz<U>() const;

    /* returns true if unallocated otherwise false */
    bool empty() const;

    size_t sizeBytes() const;

private:
    /* device pointer */
    void *data_;

    /* allocated size in bytes */
    size_t sizeBytes_;

    /* pointer to reference counter in CPU memory */
    int *refcount_;
};

}  // namespace cuda

namespace device {

using fastreg::cuda::DeviceMemory;

}  // namespace device
}  // namespace fastreg

/* inline implementations of DeviceMemory */
template <class T>
inline T *fastreg::cuda::DeviceMemory::ptr() {
    return (T *) data_;
}
template <class T>
inline const T *fastreg::cuda::DeviceMemory::ptr() const {
    return (const T *) data_;
}

template <class U>
inline fastreg::cuda::DeviceMemory::operator fastreg::cuda::PtrSz<U>() const {
    PtrSz<U> result;
    result.data = (U *) ptr<U>();
    result.size = sizeBytes_ / sizeof(U);
    return result;
}
