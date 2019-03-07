/* fastreg includes */
#include <cuda/cuda_utils.hpp>
#include <cuda/device.hpp>

__global__ void fastreg::device::clear_kernel(fastreg::device::Volume vol) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > vol.dims.x - 1 || y > vol.dims.y - 1) {
        return;
    }

    float2* beg = vol.beg(x, y);
    float2* end = beg + vol.dims.x * vol.dims.y * vol.dims.z;

    for (float2* pos = beg; pos != end; pos = vol.zstep(pos)) {
        *pos = make_float2(0.f, 0.f);
    }
}

void fastreg::device::clear(fastreg::device::Volume& vol) {
    dim3 block(64, 16);
    dim3 grid(divUp(vol.dims.x, block.x), divUp(vol.dims.y, block.y));

    clear_kernel<<<grid, block>>>(vol);
    cudaSafeCall(cudaGetLastError());
}

__global__ void fastreg::device::initialise_kernel(fastreg::device::Volume volume, float* data, short* labels_data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume.dims.x || y >= volume.dims.y) {
        return;
    }

    float2* vptr = volume.beg(x, y);
    for (int i = 0; i <= volume.dims.z - 1; ++i, vptr = volume.zstep(vptr)) {
        /* get the pixel intensity */
        float intensity = data[i * volume.dims.y * volume.dims.x + y * volume.dims.x + x];
        /* get the label */
        float label = (float) labels_data[i * volume.dims.y * volume.dims.x + y * volume.dims.x + x];

        *vptr = make_float2(intensity, label);
    }
}

void fastreg::device::initialise(fastreg::device::Volume& vol, float* data, short* labels_data) {
    dim3 block(64, 16);
    dim3 grid(divUp(vol.dims.x, block.x), divUp(vol.dims.y, block.y));

    initialise_kernel<<<grid, block>>>(vol, data, labels_data);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}
