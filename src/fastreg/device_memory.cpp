/* fastreg includes */
#include <cuda/device_memory.hpp>
#include <safe_call.hpp>

/* sys headers */
#include <cassert>
#include <cstdlib>
#include <iostream>

void fastreg::cuda::error(const char *error_string, const char *file, const int line, const char *func) {
    std::cout << "error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

/* XADD */
#ifdef __GNUC__

#if __GNUC__ * 10 + __GNUC_MINOR__ >= 42

#if !defined WIN32 && (defined __i486__ || defined __i586__ || defined __i686__ || defined __MMX__ || \
                       defined __SSE__ || defined __ppc__)
#define CV_XADD __sync_fetch_and_add
#else
#include <ext/atomicity.h>
#define CV_XADD __gnu_cxx::__exchange_and_add
#endif
#else
#include <bits/atomicity.h>
#if __GNUC__ * 10 + __GNUC_MINOR__ >= 34
#define CV_XADD __gnu_cxx::__exchange_and_add
#else
#define CV_XADD __exchange_and_add
#endif
#endif

#elif defined WIN32 || defined _WIN32
#include <intrin.h>
#define CV_XADD(addr, delta) _InterlockedExchangeAdd((long volatile *) (addr), (delta))
#else

template <typename _Tp>
static inline _Tp CV_XADD(_Tp *addr, _Tp delta) {
    int tmp = *addr;
    *addr += delta;
    return tmp;
}

#endif

/*
 * DEVICE MEMORY
 */

fastreg::cuda::DeviceMemory::DeviceMemory() : data_(0), sizeBytes_(0), refcount_(0) {}

fastreg::cuda::DeviceMemory::DeviceMemory(void *ptr_arg, size_t sizeBytes_arg)
    : data_(ptr_arg), sizeBytes_(sizeBytes_arg), refcount_(0) {}

fastreg::cuda::DeviceMemory::DeviceMemory(size_t sizeBtes_arg) : data_(0), sizeBytes_(0), refcount_(0) {
    create(sizeBtes_arg);
}
fastreg::cuda::DeviceMemory::~DeviceMemory() { release(); }

fastreg::cuda::DeviceMemory::DeviceMemory(const DeviceMemory &other_arg)
    : data_(other_arg.data_), sizeBytes_(other_arg.sizeBytes_), refcount_(other_arg.refcount_) {
    if (refcount_)
        CV_XADD(refcount_, 1);
}

fastreg::cuda::DeviceMemory &fastreg::cuda::DeviceMemory::operator=(const fastreg::cuda::DeviceMemory &other_arg) {
    if (this != &other_arg) {
        if (other_arg.refcount_)
            CV_XADD(other_arg.refcount_, 1);
        release();

        data_      = other_arg.data_;
        sizeBytes_ = other_arg.sizeBytes_;
        refcount_  = other_arg.refcount_;
    }
    return *this;
}

void fastreg::cuda::DeviceMemory::create(size_t sizeBytes_arg) {
    if (sizeBytes_arg == sizeBytes_)
        return;

    if (sizeBytes_arg > 0) {
        if (data_)
            release();

        sizeBytes_ = sizeBytes_arg;

        cudaSafeCall(cudaMalloc(&data_, sizeBytes_));

        refcount_  = new int;
        *refcount_ = 1;
    }
}

void fastreg::cuda::DeviceMemory::copyTo(DeviceMemory &other) const {
    if (empty())
        other.release();
    else {
        other.create(sizeBytes_);
        cudaSafeCall(cudaMemcpy(other.data_, data_, sizeBytes_, cudaMemcpyDeviceToDevice));
    }
}

void fastreg::cuda::DeviceMemory::release() {
    if (refcount_ && CV_XADD(refcount_, -1) == 1) {
        delete refcount_;
        cudaSafeCall(cudaFree(data_));
    }
    data_      = 0;
    sizeBytes_ = 0;
    refcount_  = 0;
}

void fastreg::cuda::DeviceMemory::upload(const void *host_ptr_arg, size_t sizeBytes_arg) {
    create(sizeBytes_arg);
    cudaSafeCall(cudaMemcpy(data_, host_ptr_arg, sizeBytes_, cudaMemcpyHostToDevice));
}

void fastreg::cuda::DeviceMemory::download(void *host_ptr_arg) const {
    cudaSafeCall(cudaMemcpy(host_ptr_arg, data_, sizeBytes_, cudaMemcpyDeviceToHost));
}

void fastreg::cuda::DeviceMemory::swap(DeviceMemory &other_arg) {
    std::swap(data_, other_arg.data_);
    std::swap(sizeBytes_, other_arg.sizeBytes_);
    std::swap(refcount_, other_arg.refcount_);
}

bool fastreg::cuda::DeviceMemory::empty() const { return !data_; }

size_t fastreg::cuda::DeviceMemory::sizeBytes() const { return sizeBytes_; }
