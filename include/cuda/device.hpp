#pragma once

/* fastreg includes */
#include <cuda/volume.hpp>
#include <safe_call.hpp>

/*
 * VOLUME
 */

__device__ __forceinline__ float2* fastreg::device::Volume::beg(int x, int y) const { return data + x + dims.x * y; }

__device__ __forceinline__ float2* fastreg::device::Volume::zstep(float2* const ptr) const {
    return ptr + dims.x * dims.y;
}

__device__ __forceinline__ float2* fastreg::device::Volume::operator()(int x, int y, int z) const {
    return data + x + y * dims.x + z * dims.y * dims.x;
}
