/* fastreg includes */
#include <cuda/cuda_utils.hpp>
#include <solver.hpp>

/*
 * DICE
 */

static void fastreg::device::calc_dice_coefficient(Volume& image_1, Volume& image_2, int no_labels) {
    int no_voxels = image_1.dims.x * image_1.dims.y * image_1.dims.z;

    float2* labels_1 = new float2[no_voxels];
    float2* labels_2 = new float2[no_voxels];

    cudaSafeCall(cudaMemcpy(labels_1, image_1.data, no_voxels * sizeof(float2), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(labels_2, image_2.data, no_voxels * sizeof(float2), cudaMemcpyDeviceToHost));

    float dice[no_labels]   = {0.f};
    float voxels[no_labels] = {0.f};

    for (int i = 0; i < image_1.dims.x - 1; i++) {
        for (int j = 0; j < image_1.dims.y - 1; j++) {
            for (int k = 0; k < image_1.dims.z - 1; k++) {
                float label_1 = labels_1[k * image_1.dims.y * image_1.dims.x + j * image_1.dims.x + i].y;
                float label_2 = labels_2[k * image_1.dims.y * image_1.dims.x + j * image_1.dims.x + i].y;

                voxels[(int) label_1] += 1.f;
                voxels[(int) label_2] += 1.f;

                if (label_1 == label_2) {
                    dice[(int) label_1] += 1.f;
                }
            }
        }
    }

    float numerator   = 0.f;
    float denominator = 0.f;

    std::cout << "DICE COEFFICIENT" << std::endl;
    for (int i = 1; i < no_labels; i++) {
        numerator += dice[i];
        denominator += voxels[i];

        std::cout << "label " << i << ": " << 2.f * dice[i] / voxels[i] << std::endl;
    }
    std::cout << "all labels: " << 2.f * numerator / denominator << std::endl;

    delete labels_1, labels_2;
}

/*
 * POTENTIAL GRADIENT
 */

__global__ void fastreg::device::calculate_potential_gradient_kernel(float2* phi_n, float2* phi_global_o_psi_inv,
                                                                     float4* nabla_phi_n, Mat4f* J_inv,
                                                                     float4* L_o_psi_inv, float4* nabla_U, float w_reg,
                                                                     int dim_x, int dim_y, int dim_z) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x == 0 || x >= dim_x - 1 || y == 0 || y >= dim_y - 1) {
        return;
    }

#pragma unroll
    for (int i = 1; i <= dim_z - 2; ++i) {
        int idx = get_global_idx(x, y, i, dim_x, dim_y, dim_z);

        float intensity_n                = phi_n[idx].x;
        float intensity_global_o_psi_inv = phi_global_o_psi_inv[idx].x;

        float determinant = det(J_inv[idx]);

        float4 functional_gradient =
            ((intensity_n - intensity_global_o_psi_inv) * nabla_phi_n[idx] + w_reg * L_o_psi_inv[idx]) * determinant;
        nabla_U[idx] = functional_gradient;
    }
}

void fastreg::device::calculate_potential_gradient(fastreg::device::Volume& phi_n,
                                                   fastreg::device::Volume& phi_global_o_psi_inv,
                                                   fastreg::device::VolumeGradient& nabla_phi_n,
                                                   fastreg::device::Jacobian& J_inv,
                                                   fastreg::device::Laplacian& L_o_psi_inv,
                                                   fastreg::device::PotentialGradient& nabla_U, float w_reg) {
    dim3 block(64, 16);
    dim3 grid(divUp(phi_n.dims.x, block.x), divUp(phi_n.dims.y, block.y));

    calculate_potential_gradient_kernel<<<grid, block>>>(phi_n.data, phi_global_o_psi_inv.data, nabla_phi_n.data,
                                                         J_inv.data, L_o_psi_inv.data, nabla_U.data, w_reg,
                                                         phi_n.dims.x, phi_n.dims.y, phi_n.dims.z);
    cudaSafeCall(cudaGetLastError());
}

/*
 * DIVERGENCE
 */

__global__ void fastreg::device::calc_divergence_kernel(float* rho, float4* v, float* div, int dim_x, int dim_y,
                                                        int dim_z) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x == 0 || x >= dim_x - 1 || y == 0 || y >= dim_y - 1) {
        return;
    }

    int idx_x_1 = x + 1;
    int idx_x_2 = x - 1;

    int idx_y_1 = y + 1;
    int idx_y_2 = y - 1;

#pragma unroll
    for (int i = 1; i <= dim_z - 2; ++i) {
        int idx_z_1 = i + 1;
        int idx_z_2 = i - 1;

        float4 vel = v[get_global_idx(x, y, i, dim_x, dim_y, dim_z)];
        float p    = rho[get_global_idx(x, y, i, dim_x, dim_y, dim_z)];

        float term11 = (vel.x > 0.f) ? -vel.x * p : -vel.x * rho[get_global_idx(idx_x_1, y, i, dim_x, dim_y, dim_z)];
        float term12 = (v[get_global_idx(idx_x_2, y, i, dim_x, dim_y, dim_z)].x > 0.f)
                           ? v[get_global_idx(idx_x_2, y, i, dim_x, dim_y, dim_z)].x *
                                 rho[get_global_idx(idx_x_2, y, i, dim_x, dim_y, dim_z)]
                           : v[get_global_idx(idx_x_2, y, i, dim_x, dim_y, dim_z)].x * p;

        float term21 = (vel.y > 0.f) ? -vel.y * p : -vel.y * rho[get_global_idx(x, idx_y_1, i, dim_x, dim_y, dim_z)];
        float term22 = (v[get_global_idx(x, idx_y_2, i, dim_x, dim_y, dim_z)].y > 0.f)
                           ? v[get_global_idx(x, idx_y_2, i, dim_x, dim_y, dim_z)].y *
                                 rho[get_global_idx(x, idx_y_2, i, dim_x, dim_y, dim_z)]
                           : v[get_global_idx(x, idx_y_2, i, dim_x, dim_y, dim_z)].y * p;

        float term31 = (vel.z > 0.f) ? -vel.z * p : -vel.z * rho[get_global_idx(x, y, idx_z_1, dim_x, dim_y, dim_z)];
        float term32 = (v[get_global_idx(x, y, idx_z_2, dim_x, dim_y, dim_z)].z > 0.f)
                           ? v[get_global_idx(x, y, idx_z_2, dim_x, dim_y, dim_z)].z *
                                 rho[get_global_idx(x, y, idx_z_2, dim_x, dim_y, dim_z)]
                           : v[get_global_idx(x, y, idx_z_2, dim_x, dim_y, dim_z)].z * p;

        div[get_global_idx(x, y, i, dim_x, dim_y, dim_z)] = term11 + term12 + term21 + term22 + term31 + term32;
    }
}

void fastreg::device::calc_divergence(fastreg::device::Density& rho, fastreg::device::Velocity& v,
                                      fastreg::device::Divergence& div) {
    dim3 block(64, 16);
    dim3 grid(divUp(rho.dims.x, block.x), divUp(rho.dims.y, block.y));

    calc_divergence_kernel<<<grid, block>>>(rho.data, v.data, div.data, rho.dims.x, rho.dims.y, rho.dims.z);
    cudaSafeCall(cudaGetLastError());
}

/*
 * DENSITY PDE
 */

__global__ void fastreg::device::update_density_kernel(float* rho, float* div, float alpha, int dim_x, int dim_y,
                                                       int dim_z) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x == 0 || x >= dim_x - 1 || y == 0 || y >= dim_y - 1) {
        return;
    }

#pragma unroll
    for (int i = 1; i <= dim_z - 2; ++i) {
        int idx = get_global_idx(x, y, i, dim_x, dim_y, dim_z);
        rho[idx] += alpha * div[idx];
    }
}

void fastreg::device::update_density(fastreg::device::Density& rho, fastreg::device::Divergence& div, float alpha) {
    /* integrate in time */
    dim3 block(64, 16);
    dim3 grid(divUp(rho.dims.x, block.x), divUp(rho.dims.y, block.y));

    update_density_kernel<<<grid, block>>>(rho.data, div.data, alpha, rho.dims.x, rho.dims.y, rho.dims.z);
    cudaSafeCall(cudaGetLastError());
}

/*
 * VELOCITY PDE
 */

__global__ void fastreg::device::update_velocity_kernel(float4* v, float4* dv_v, float4* delta_v, float* rho,
                                                        float4* nabla_U, float alpha, float t, float tau, float p,
                                                        float C, int dim_x, int dim_y, int dim_z) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x == 0 || x >= dim_x - 1 || y == 0 || y >= dim_y - 1) {
        return;
    }

    float multiplier_1 = p + 1.f;
    float multiplier_2 = C * powf(p, 2.f) * powf(t, p - 2.f);

#pragma unroll
    for (int i = 1; i <= dim_z - 2; ++i) {
        int idx = get_global_idx(x, y, i, dim_x, dim_y, dim_z);

        float4 friction  = multiplier_1 / t * v[idx];
        float4 diffusion = tau * delta_v[idx];
        float4 potential = multiplier_2 / (fabs(rho[idx]) + 1e-5f) * nabla_U[idx];

        v[idx] -= alpha * (friction + dv_v[idx] - diffusion + potential);
    }
}

void fastreg::device::update_velocity(fastreg::device::Velocity& v, fastreg::device::DVV& dv_v,
                                      fastreg::device::Laplacian& delta_v, fastreg::device::Density& rho,
                                      fastreg::device::DeformationField& nabla_U, float alpha, float t, float tau,
                                      float p, float C) {
    dim3 block(64, 16);
    dim3 grid(divUp(v.dims.x, block.x), divUp(v.dims.y, block.y));

    /* integrate in time */
    update_velocity_kernel<<<grid, block>>>(v.data, dv_v.data, delta_v.data, rho.data, nabla_U.data, alpha, t, tau, p,
                                            C, v.dims.x, v.dims.y, v.dims.z);
    cudaSafeCall(cudaGetLastError());
}

/*
 * DEFORMATION FIELD PDE
 */

__global__ void fastreg::device::update_psi_kernel(float4* psi, float4* v, float4* updates, float alpha, int dim_x,
                                                   int dim_y, int dim_z) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x == 0 || x >= dim_x - 1 || y == 0 || y >= dim_y - 1) {
        return;
    }

#pragma unroll
    for (int i = 1; i <= dim_z - 2; ++i) {
        int idx       = get_global_idx(x, y, i, dim_x, dim_y, dim_z);
        float4 update = alpha * interpolate_field(v, trunc(psi[idx]), dim_x, dim_y, dim_z);

        psi[idx] += update;
        updates[idx] = update;
    }
}

void fastreg::device::update_psi(fastreg::device::DeformationField& psi, fastreg::device::Velocity& v, float4* updates,
                                 float alpha) {
    /* integrate in time */
    dim3 block(64, 16);
    dim3 grid(divUp(psi.dims.x, block.x), divUp(psi.dims.y, block.y));

    update_psi_kernel<<<grid, block>>>(psi.data, v.data, updates, alpha, v.dims.x, v.dims.y, v.dims.z);
    cudaSafeCall(cudaGetLastError());
}

/*
 * INVERSE DEFORMATION FIELD PDE
 */

__global__ void fastreg::device::update_psi_inv_kernel(float4* psi_inv, Mat4f* J_psi_inv, float4* v, float alpha,
                                                       int dim_x, int dim_y, int dim_z) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x == 0 || x >= dim_x - 1 || y == 0 || y >= dim_y - 1) {
        return;
    }

#pragma unroll
    for (int i = 1; i <= dim_z - 2; ++i) {
        int idx       = get_global_idx(x, y, i, dim_x, dim_y, dim_z);
        float4 update = alpha * (J_psi_inv[idx] * v[idx]);

        psi_inv[idx] -= update;
    }
}

void fastreg::device::update_psi_inv(fastreg::device::DeformationField& psi_inv, fastreg::device::Jacobian& J_psi_inv,
                                     fastreg::device::Velocity& v, float alpha) {
    /* integrate in time */
    dim3 block(64, 16);
    dim3 grid(divUp(psi_inv.dims.x, block.x), divUp(psi_inv.dims.y, block.y));

    update_psi_inv_kernel<<<grid, block>>>(psi_inv.data, J_psi_inv.data, v.data, alpha, v.dims.x, v.dims.y, v.dims.z);
    cudaSafeCall(cudaGetLastError());
}

/*
 * PIPELINE
 */

void fastreg::device::estimate_psi(SDFs& sdfs, fastreg::device::DeformationField& psi,
                                   fastreg::device::DeformationField& psi_inv, fastreg::device::Velocity& v,
                                   fastreg::device::Density& rho, fastreg::device::Divergence& div,
                                   fastreg::device::SpatialGradients* spatial_grads, Differentiators& differentiators,
                                   fastreg::device::Reductor* r, SolverParams& params) {
    /* create cuda streams */
    int no_streams = 5;
    cudaStream_t streams[no_streams];
    for (int i = 0; i < no_streams; i++) {
        cudaSafeCall(cudaStreamCreate(&streams[i]));
    }

    /* calculate the gradient of phi_n */
    differentiators.volume_diff.calculate(*(spatial_grads->nabla_phi_n));

    /* run the solver */
    int3 dims = psi.dims;

    dim3 block(32, 16);
    dim3 grid(divUp(dims.x, block.x), divUp(dims.y, block.y));

    /* init density */
    init_density_kernel<<<grid, block, 0, streams[0]>>>(rho, params.rho_0);
    cudaSafeCall(cudaGetLastError());

    float2 curr_max_update_norm;

    float e_prev = std::numeric_limits<float>::infinity();
    float e_curr = std::numeric_limits<float>::infinity();

    int iter = 1;

    float alpha = params.alpha;
    float t     = alpha;
    while (iter <= params.max_iter) {
        if (iter == 1 || iter % 100 == 0) {
            std::cout << "iter. no. " << iter << std::endl;
        }

        /* calculate div(rho * v) and update rho */
        calc_divergence_kernel<<<grid, block, 0, streams[0]>>>(rho.data, v.data, div.data, dims.x, dims.y, dims.z);
        cudaSafeCall(cudaGetLastError());
        update_density_kernel<<<grid, block, 0, streams[0]>>>(rho.data, div.data, alpha, dims.x, dims.y, dims.z);
        cudaSafeCall(cudaGetLastError());

        /* calculate the jacobian of psi */
        estimate_jacobian_kernel<<<grid, block, 0, streams[1]>>>(differentiators.diff_inv, *(spatial_grads->J_inv));
        cudaSafeCall(cudaGetLastError());
        /* calculate the laplacian of psi */
        estimate_laplacian_kernel<<<grid, block, 0, streams[2]>>>(differentiators.second_order_diff,
                                                                  *(spatial_grads->L));
        cudaSafeCall(cudaGetLastError());
        /* interpolate the laplacian of psi at psi_inv */
        interpolate_laplacian_kernel<<<grid, block, 0, streams[2]>>>(*(spatial_grads->L), *(spatial_grads->L_o_psi_inv),
                                                                     psi_inv);
        cudaSafeCall(cudaGetLastError());

        /* calculate current value of the energy functional */
        e_prev = e_curr;

        if ((params.verbosity == 1 && (iter == 1 || iter % 100 == 0 || iter == params.max_iter) ||
             params.verbosity == 2)) {
            apply_kernel<<<grid, block, 0, streams[3]>>>(sdfs.phi_n, sdfs.phi_n_psi, psi);
            cudaSafeCall(cudaGetLastError());
            estimate_deformation_jacobian_kernel<<<grid, block, 0, streams[4]>>>(differentiators.diff,
                                                                                 *(spatial_grads->J));
            cudaSafeCall(cudaGetLastError());

            e_curr = r->energy(sdfs.phi_global.data, sdfs.phi_n_psi.data, spatial_grads->J->data, params.w_reg, true);
        }

        if (std::isnan(e_curr)) {
            break;
        }

        /*
         * PDE'S
         */

        /* calculate gradient of the potential */
        fastreg::device::calculate_potential_gradient(
            sdfs.phi_n, sdfs.phi_global_psi_inv, *(spatial_grads->nabla_phi_n), *(spatial_grads->J_inv),
            *(spatial_grads->L_o_psi_inv), *(spatial_grads->nabla_U), params.w_reg);
        cudaSafeCall(cudaGetLastError());

        /* calculate dv_v */
        calc_dv_v_kernel<<<grid, block, 0, streams[0]>>>(v, *(spatial_grads->dv_v));
        cudaSafeCall(cudaGetLastError());
        /* calculate delta_v */
        estimate_laplacian_velocity_kernel<<<grid, block, 0, streams[1]>>>(differentiators.velocity_second_order_diff,
                                                                           *(spatial_grads->delta_v));
        cudaSafeCall(cudaGetLastError());

        /* update v */
        fastreg::device::update_velocity(v, *(spatial_grads->dv_v), *(spatial_grads->delta_v), rho,
                                         *(spatial_grads->nabla_U), alpha, t, params.tau, params.p, params.C);
        cudaSafeCall(cudaGetLastError());

        /* update psi */
        update_psi_kernel<<<grid, block, 0, streams[0]>>>(psi.data, v.data, r->updates, alpha, dims.x, dims.y, dims.z);
        cudaSafeCall(cudaGetLastError());
        /* update psi_inv */
        update_psi_inv_kernel<<<grid, block, 0, streams[1]>>>(psi_inv.data, spatial_grads->J_inv->data, v.data, alpha,
                                                              dims.x, dims.y, dims.z);
        cudaSafeCall(cudaGetLastError());

        /* apply psi_inv to phi_global */
        apply_kernel<<<grid, block, 0, streams[1]>>>(sdfs.phi_global, sdfs.phi_global_psi_inv, psi_inv);
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaStreamSynchronize(streams[1]));

        /* get value of the max. update norm at the current iteration of the solver */
        curr_max_update_norm = r->max_update_norm();
        if ((params.verbosity == 1 && (iter == 1 || iter % 100 == 0 || iter == params.max_iter) ||
             params.verbosity == 2)) {
            int idx_x = curr_max_update_norm.y / (psi.dims.x * psi.dims.y);
            int idx_y = (curr_max_update_norm.y - idx_x * psi.dims.x * psi.dims.y) / psi.dims.x;
            int idx_z = curr_max_update_norm.y - psi.dims.x * (idx_y + psi.dims.y * idx_x);

            std::cout << "max. update norm " << curr_max_update_norm.x << " at voxel (" << idx_z << ", " << idx_y
                      << ", " << idx_x << ")" << std::endl;
        }

        if (curr_max_update_norm.x <= params.max_update_norm) {
            std::cout << "SOLVER CONVERGED AFTER " << iter << " ITERATIONS" << std::endl;
            break;
        }

        if (iter == params.max_iter) {
            std::cout << "SOLVER REACHED MAX. NO. OF ITERATIONS WITHOUT CONVERGING" << std::endl;
            break;
        }

        iter++;
        t += alpha;
        if (params.reset_iter > 0 && iter % params.reset_iter == 0) {
            t = params.alpha;
        }
    }

    apply_kernel<<<grid, block>>>(sdfs.phi_n, sdfs.phi_n_psi, psi);
    cudaSafeCall(cudaGetLastError());

    for (int i = 0; i < no_streams; i++) {
        cudaSafeCall(cudaStreamDestroy(streams[i]));
    }

    std::cout << "POST-REGISTRATION" << std::endl;
    fastreg::device::calc_dice_coefficient(sdfs.phi_n_psi, sdfs.phi_global, params.no_labels);
}
