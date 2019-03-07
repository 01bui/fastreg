#pragma once

/* fastreg includes */
#include <cuda/device_memory.hpp>
#include <params.hpp>
#include <precomp.hpp>
#include <types.hpp>

/* sys headers */
#include <iostream>

namespace fastreg {
namespace cuda {

/*
 * VOLUME
 */

class Volume {
public:
    /* constructor */
    Volume(cv::Vec3i dims_);
    /* destructor */
    ~Volume();

    /* get dimensions of the volume */
    cv::Vec3i get_dims() const;

    /* get the data in the volume */
    CudaData get_data();
    const CudaData get_data() const;
    /* set the data in the volume */
    void set_data(CudaData &data);

    /* initialise the volume with intensity values and labels */
    void initialise(float *image_data, short *labels_data);

    /* clear volume */
    void clear();
    /* print volume */
    void print();

protected:
    CudaData data;  /* volume data */
    cv::Vec3i dims; /* volume dimensions */
};

}  // namespace cuda
}  // namespace fastreg

namespace fastreg {
namespace device {

struct Volume {
    /* constructor */
    Volume(float2 *const data_, const int3 dims_) : data(data_), dims(dims_) {}

    __device__ __forceinline__ float2 *beg(int x, int y) const;
    __device__ __forceinline__ float2 *zstep(float2 *const ptr) const;
    __device__ __forceinline__ float2 *operator()(int x, int y, int z) const;

    float2 *const data; /* volume data */
    const int3 dims;    /* volume dimensions */
};

/* clear volume */
__global__ void clear_kernel(Volume vol);
void clear(Volume &vol);

/* initialise intensities and labels */
__global__ void initialise_kernel(Volume vol, float *data, short *labels_data);
void initialise(Volume &vol, float *data, short *labels_data);

}  // namespace device
}  // namespace fastreg
