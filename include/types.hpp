#pragma once

/* cuda includes */
#include <vector_functions.h>

/* fastreg includes */
#include <cuda/device_memory.hpp>

/* opencv includes */
#include <opencv2/core/core.hpp>

typedef cv::Vec3i Vec3i;
typedef fastreg::cuda::DeviceMemory CudaData;

namespace fastreg {

struct ScopeTime {
    const char *name;
    double start;
    ScopeTime(const char *name);
    ~ScopeTime();
};

struct SampledScopeTime {
public:
    enum { EACH = 34 };
    SampledScopeTime(double &time_ms);
    ~SampledScopeTime();

private:
    double getTime();
    SampledScopeTime(const SampledScopeTime &);
    SampledScopeTime &operator=(const SampledScopeTime &);

    double &time_ms_;
    double start;
};

}  // namespace fastreg
