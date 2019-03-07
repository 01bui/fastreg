#pragma once

/* fastreg includes */
#include <cuda/device_memory.hpp>
#include <cuda/volume.hpp>

#include <params.hpp>
#include <precomp.hpp>

/* sys headers */
#include <iostream>

namespace fastreg {
namespace cuda {

/*
 * SCALAR FIELD
 */

class ScalarField {
public:
    /* constructor */
    ScalarField(cv::Vec3i dims_);
    /* destructor */
    ~ScalarField();

    /* get field data*/
    CudaData get_data();
    const CudaData get_data() const;

    /* get dims */
    int3 get_dims();

    /* clear field */
    void clear();
    /* sum values in the field */
    float sum();
    /* print field */
    void print();

private:
    CudaData data; /* field data */
    int3 dims;     /* field dimensions */
};

/*
 * DENSITY
 */

class Density : public ScalarField {
public:
    /* constructor */
    Density(cv::Vec3i dims_, float rho_0_);
    /* destructor */
    ~Density();

    void init(fastreg::cuda::Volume &phi, float rho_0);
    void clear(float rho_0);
};

/*
 * DIVERGENCE
 */

typedef ScalarField Divergence;

}  // namespace cuda

namespace device {

/*
 * SCALAR FIELD
 */

struct ScalarField {
    /* constructor */
    ScalarField(float *const data_, int3 dims_) : data(data_), dims(dims_) {}

    __device__ __forceinline__ float *beg(int x, int y) const;
    __device__ __forceinline__ float *zstep(float *const ptr) const;

    __device__ __forceinline__ float *operator()(int idx) const;
    __device__ __forceinline__ float *operator()(int x, int y, int z) const;

    float *const data; /* field data */
    const int3 dims;
};

/* clear */
__global__ void clear_kernel(fastreg::device::ScalarField field);
void clear(ScalarField &field);

/*
 * DENSITY AND DIVERGENCE
 */

typedef ScalarField Density;
typedef ScalarField Divergence;

/* clear divergence */
__global__ void init_density_kernel(fastreg::device::Density rho, float rho_0);
void init_density(Density &rho, float &rho_0);

__global__ void init_density_kernel(fastreg::device::Volume phi, fastreg::device::Density rho, float rho_0);
void init_density(Volume &phi, Density &rho, float rho_0);

/* sum */
float reduce_sum(fastreg::device::ScalarField &field);

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce_sum_kernel(float *g_idata, float *g_odata, unsigned int n);
float final_reduce_sum(float *d_odata, int numBlocks);

}  // namespace device
}  // namespace fastreg
