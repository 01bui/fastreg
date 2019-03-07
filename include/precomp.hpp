#pragma once

/* cuda includes */
#include <cuda.h>

/* sys headers */
#include <stdio.h>

namespace fastreg {

template <typename D, typename S>
inline D device_cast(const S &source) {
    return *reinterpret_cast<const D *>(source.val);
}

}  // namespace fastreg

/*
 * checks if x is a power of 2
 */

bool isPow2(unsigned int x);

/*
 * computes the nearest power of 2 larger than x
 */

int nextPow2(int x);

/*
 * compute the number of threads and blocks to use for the given reduction kernel; we set threads
 * block to the minimum of maxThreads and n/2; we observe the maximum specified number of blocks, because each thread
 * in the kernel can process a variable number of elements
 */

void get_num_blocks_and_threads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads);
