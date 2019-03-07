#pragma once

/* fastreg incldues */
#include <cuda/vector_fields.hpp>
#include <precomp.hpp>
#include <safe_call.hpp>

namespace fastreg {
namespace device {

/*
 * REDUCTOR
 */

class Reductor {
public:
    /* constructor */
    Reductor(int3 dims_);
    /* destructor */
    ~Reductor();

    /* optical flow */
    float energy(float2 *phi_global_data, float2 *phi_n_psi_data, Mat4f *J_data, float w_reg, float print = false);

    /* max. */
    float2 max_update_norm();
    float2 voxel_max_energy(float2 *phi_global_data, float2 *phi_n_data, Mat4f *J_data, float w_reg);

    float4 *updates;

private:
    /* data term */
    void data_energy(float2 *phi_global_data, float2 *phi_n_data);
    /* regularisation term */
    void reg_energy(Mat4f *J_data);

    int3 dims;
    float vsz, trunc_dist;
    int no_voxels;

    /* streams to calculate data and regularisation energies in parallel */
    cudaStream_t streams[4];
    /* no. of blocks and threads for the reductions used to calculate value of the energy functional */
    int blocks, threads;

    float *h_data_out, *d_data_out;
    float *h_reg_out, *d_reg_out;
    float *h_reg_level_set_out, *d_reg_level_set_out;

    float2 *h_max_out, *d_max_out;
};

/*
 * DATA TERM
 */

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce_data_kernel(float2 *g_idata_global, float2 *g_idata_n, float *g_odata, unsigned int n);
void reduce_data(int size, int threads, int blocks, float2 *d_idata_global, float2 *d_idata_n, float *d_odata,
                 cudaStream_t &stream);

/*
 * REGULARISATION TERM
 */

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce_reg_kernel(Mat4f *g_idata, float *g_odata, unsigned int n);
void reduce_reg(int size, int threads, int blocks, Mat4f *d_idata, float *d_odata, cudaStream_t &stream);

/*
 * MAX.
 */

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce_max_kernel(float4 *updates, float2 *g_o_max_data, unsigned int n);
void reduce_max(int size, int threads, int blocks, float4 *updates, float2 *d_o_max_data, cudaStream_t &stream);

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce_voxel_max_energy_kernel(float2 *d_idata_global, float2 *d_idata_n, Mat4f *d_idata_reg,
                                               float2 *d_o_data, float w_reg, unsigned int n);
void reduce_voxel_max_energy(int size, int threads, int blocks, float2 *d_idata_global, float2 *d_idata_n,
                             Mat4f *d_idata_reg, float w_reg, float2 *d_odata, cudaStream_t &stream);

/*
 * CPU FINAL REDUCTIONS
 */

float final_reduce(float *h_odata, float *d_odata, int numBlocks);
float2 final_reduce_max(float2 *h_o_max_data, float2 *d_o_max_data, int numBlocks, int3 dims);

}  // namespace device
}  // namespace fastreg
