#pragma once

/* opencv includes */
#include <opencv2/core/core.hpp>

/* parameters */
struct Params {
    cv::Vec3i volume_dims; /* volume dimensions in voxels */
    cv::Vec3f volume_size; /* volume size in metres */

    int reset_iter = -1; /* iteration when to reset t in order to avoid underdamping */
    int verbosity  = 0;  /* solver verbosity */

    int no_labels /* no. of segmentations */;
    int max_iter /* max. no of iterations of the solver */;
    float max_update_norm /* max. update norm */, tau /* velocity diffusion coefficient */, rho_0 /* mass density */,
        alpha /* gradient descent step size */, w_reg /* weight of the regularisation term */, p,
        C /* weighing factors for momentum and potential energy */;

    int no_voxels() { return volume_dims[0] * volume_dims[1] * volume_dims[2]; }

    cv::Vec3f voxel_sizes() {
        return cv::Vec3f(volume_size[0] / volume_dims[0], volume_size[1] / volume_dims[1],
                         volume_size[2] / volume_dims[2]);
    }
};
