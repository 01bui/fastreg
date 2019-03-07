/* fastreg incldues */
#include <cuda/reductor.hpp>

/*
 * REDUCTOR
 */

fastreg::device::Reductor::Reductor(int3 dims_) {
    dims = dims_;

    no_voxels = dims.x * dims.y * dims.z;
    get_num_blocks_and_threads(no_voxels, 65536, 512, blocks, threads);

    h_data_out          = new float[blocks];
    h_reg_out           = new float[blocks];
    h_reg_level_set_out = new float[blocks];
    h_max_out           = new float2[blocks];

    cudaSafeCall(cudaMalloc((void **) &d_data_out, blocks * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **) &d_reg_out, blocks * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **) &d_reg_level_set_out, blocks * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **) &d_max_out, blocks * sizeof(float2)));
    cudaSafeCall(cudaMalloc((void **) &updates, no_voxels * sizeof(float4)));

    /* cuda streams */
    for (int i = 0; i < 4; i++) {
        cudaSafeCall(cudaStreamCreate(&streams[i]));
    }
}

fastreg::device::Reductor::~Reductor() {
    delete h_data_out, h_reg_out, h_reg_level_set_out, h_max_out;

    cudaSafeCall(cudaFree(d_data_out));
    cudaSafeCall(cudaFree(d_reg_out));
    cudaSafeCall(cudaFree(d_reg_level_set_out));
    cudaSafeCall(cudaFree(d_max_out));
    cudaSafeCall(cudaFree(updates));

    for (int i = 0; i < 4; i++) {
        cudaSafeCall(cudaStreamDestroy(streams[i]));
    }
}

float fastreg::device::Reductor::energy(float2 *phi_global_data, float2 *phi_n_psi_data, Mat4f *J_data, float w_reg,
                                        float print) {
    data_energy(phi_global_data, phi_n_psi_data);
    reg_energy(J_data);

    cudaSafeCall(cudaStreamSynchronize(streams[0]));
    float e_data = 0.5f * final_reduce(h_data_out, d_data_out, blocks);
    cudaSafeCall(cudaStreamSynchronize(streams[1]));
    float e_reg = 0.5f * final_reduce(h_reg_out, d_reg_out, blocks);

    float val = e_data + w_reg * e_reg;
    if (print) {
        std::cout << "data energy + w_reg * reg energy = " << e_data << " + " << w_reg << " * " << e_reg << " = " << val
                  << std::endl;
    }

    return val;
}

void fastreg::device::Reductor::data_energy(float2 *phi_global_data, float2 *phi_n_data) {
    reduce_data(no_voxels, threads, blocks, phi_global_data, phi_n_data, d_data_out, streams[0]);
    cudaSafeCall(cudaGetLastError());
}

void fastreg::device::Reductor::reg_energy(Mat4f *J_data) {
    reduce_reg(no_voxels, threads, blocks, J_data, d_reg_out, streams[1]);
    cudaSafeCall(cudaGetLastError());
}

float2 fastreg::device::Reductor::max_update_norm() {
    reduce_max(no_voxels, threads, blocks, updates, d_max_out, streams[3]);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaStreamSynchronize(streams[3]));

    return final_reduce_max(h_max_out, d_max_out, blocks, dims);
}

float2 fastreg::device::Reductor::voxel_max_energy(float2 *phi_global_data, float2 *phi_n_data, Mat4f *J_data,
                                                   float w_reg) {
    reduce_voxel_max_energy(no_voxels, threads, blocks, phi_global_data, phi_n_data, J_data, w_reg, d_max_out,
                            streams[3]);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaStreamSynchronize(streams[3]));

    return final_reduce_max(h_max_out, d_max_out, blocks, dims);
}

/* sum on cpu partial sums from each block */
float fastreg::device::final_reduce(float *h_odata, float *d_odata, int numBlocks) {
    /* copy result from device to host */
    cudaSafeCall(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaGetLastError());

    float result = 0.f;
    for (int i = 0; i < numBlocks; i++) {
        result += h_odata[i];
    }

    return result;
}

/* max. of max.'s from each block */
float2 fastreg::device::final_reduce_max(float2 *h_o_max_data, float2 *d_o_max_data, int numBlocks, int3 dims) {
    /* copy result from device to host */
    cudaSafeCall(cudaMemcpy(h_o_max_data, d_o_max_data, numBlocks * sizeof(float2), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaGetLastError());

    float2 result = make_float2(0.f, 0.f);
    for (int i = 0; i < numBlocks; i++) {
        if (h_o_max_data[i].x > result.x) {
            result = h_o_max_data[i];
        }
    }

    return result;
}
