/* fastreg includes */
#include <cuda/device.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/vector_fields.hpp>

#include <safe_call.hpp>

/*
 * VECTOR FIELD
 */

__device__ __forceinline__ float4* fastreg::device::VectorField::beg(int x, int y) const {
    return data + x + dims.x * y;
}

__device__ __forceinline__ float4* fastreg::device::VectorField::zstep(float4* const ptr) const {
    return ptr + dims.x * dims.y;
}

__device__ __forceinline__ float4* fastreg::device::VectorField::operator()(int x, int y, int z) const {
    return data + x + y * dims.x + z * dims.y * dims.x;
}

__device__ __forceinline__ float4 fastreg::device::VectorField::get_displacement(int x, int y, int z) const {
    return *(data + z * dims.y * dims.x + y * dims.x + x) - make_float4((float) x, (float) y, (float) z, 0.f);
}

void fastreg::device::clear(VectorField& field) {
    dim3 block(64, 16);
    dim3 grid(divUp(field.dims.x, block.x), divUp(field.dims.y, block.y));

    clear_kernel<<<grid, block>>>(field);
    cudaSafeCall(cudaGetLastError());
}

__global__ void fastreg::device::clear_kernel(fastreg::device::VectorField field) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > field.dims.x - 1 || y > field.dims.y - 1) {
        return;
    }

    float4* beg = field.beg(x, y);
    float4* end = beg + field.dims.x * field.dims.y * field.dims.z;

    for (float4* pos = beg; pos != end; pos = field.zstep(pos)) {
        *pos = make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

/*
 * DEFORMATION FIELD
 */

void fastreg::device::init_identity(fastreg::device::DeformationField& psi) {
    dim3 block(64, 16);
    dim3 grid(divUp(psi.dims.x, block.x), divUp(psi.dims.y, block.y));

    init_identity_kernel<<<grid, block>>>(psi);
    cudaSafeCall(cudaGetLastError());
}

__global__ void fastreg::device::init_identity_kernel(fastreg::device::DeformationField psi) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi.dims.x - 1 || y > psi.dims.y - 1) {
        return;
    }

    float4 idx   = make_float4((float) x, (float) y, 0.f, 0.f);
    float4 zstep = make_float4(0.f, 0.f, 1.f, 0.f);

    float4* pos = psi.beg(x, y);
    for (int i = 0; i <= psi.dims.z - 1; idx += zstep, pos = psi.zstep(pos), ++i) {
        *pos = idx;
    }
}

__global__ void fastreg::device::apply_kernel(const Volume phi, Volume phi_warped,
                                              const fastreg::device::DeformationField psi) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > phi_warped.dims.x - 1 || y > phi_warped.dims.y - 1) {
        return;
    }

    float4* psi_ptr        = psi.beg(x, y);
    float2* phi_warped_ptr = phi_warped.beg(x, y);
    for (int i   = 0; i <= phi_warped.dims.z - 1;
         psi_ptr = psi.zstep(psi_ptr), phi_warped_ptr = phi_warped.zstep(phi_warped_ptr), ++i) {
        float4 psi_val = *psi_ptr;

        float2 intensity_deformed = interpolate_intensity(phi, trunc(psi_val));
        *phi_warped_ptr           = intensity_deformed;
    }
}

void fastreg::device::apply(const Volume& phi, Volume& phi_warped, const fastreg::device::DeformationField& psi) {
    dim3 block(64, 16);
    dim3 grid(divUp(phi.dims.x, block.x), divUp(phi.dims.y, block.y));

    apply_kernel<<<grid, block>>>(phi, phi_warped, psi);
    cudaSafeCall(cudaGetLastError());
}

__global__ void fastreg::device::estimate_inverse_kernel(fastreg::device::DeformationField psi,
                                                         fastreg::device::DeformationField psi_inv) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi_inv.dims.x - 1 || y > psi_inv.dims.y - 1) {
        return;
    }

    float4* psi_inv_ptr = psi_inv.beg(x, y);
    for (int i = 0; i <= psi_inv.dims.z - 1; psi_inv_ptr = psi_inv.zstep(psi_inv_ptr), ++i) {
        float4 psi_inv_val = *psi_inv_ptr;
        *psi_inv_ptr       = -1.f * interpolate_field(psi, trunc(psi_inv_val));
    }
}

void fastreg::device::estimate_inverse(fastreg::device::DeformationField& psi,
                                       fastreg::device::DeformationField& psi_inverse) {
    dim3 block(64, 16);
    dim3 grid(divUp(psi_inverse.dims.x, block.x), divUp(psi_inverse.dims.y, block.y));

    /* estimate inverse */
    for (int iter = 0; iter < 48; ++iter) {
        estimate_inverse_kernel<<<grid, block>>>(psi, psi_inverse);
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());
    }
}

/*
 * VOLUME DIFFERENTIATOR METHODS
 */

__global__ void fastreg::device::estimate_gradient_kernel(const fastreg::device::VolumeDifferentiator diff,
                                                          fastreg::device::VolumeGradient grad) {
    diff(grad);
}

void fastreg::device::VolumeDifferentiator::calculate(fastreg::device::VolumeGradient& grad) {
    dim3 block(64, 16);
    dim3 grid(divUp(grad.dims.x, block.x), divUp(grad.dims.y, block.y));

    estimate_gradient_kernel<<<grid, block>>>(*this, grad);
    cudaSafeCall(cudaGetLastError());
}

__device__ __forceinline__ void fastreg::device::VolumeDifferentiator::operator()(
    fastreg::device::VolumeGradient& grad) const {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x == 0 || x >= vol.dims.x - 1 || y == 0 || y >= vol.dims.y - 1) {
        return;
    }

    int idx_x_1 = x + 1;
    int idx_x_2 = x - 1;

    int idx_y_1 = y + 1;
    int idx_y_2 = y - 1;

    float4* grad_ptr = grad.beg(x, y) + vol.dims.x * vol.dims.y;

#pragma unroll
    for (int i = 1; i <= vol.dims.z - 2; grad_ptr = grad.zstep(grad_ptr), ++i) {
        int idx_z_1 = i + 1;
        int idx_z_2 = i - 1;

        float Fx1 = (*vol(idx_x_1, y, i)).x;
        float Fx2 = (*vol(idx_x_2, y, i)).x;
        float n_x = __fdividef(Fx1 - Fx2, 2.f);

        float Fy1 = (*vol(x, idx_y_1, i)).x;
        float Fy2 = (*vol(x, idx_y_2, i)).x;
        float n_y = __fdividef(Fy1 - Fy2, 2.f);

        float Fz1 = (*vol(x, y, idx_z_1)).x;
        float Fz2 = (*vol(x, y, idx_z_2)).x;
        float n_z = __fdividef(Fz1 - Fz2, 2.f);

        float4 n  = make_float4(n_x, n_y, n_z, 0.f);
        *grad_ptr = n;
    }
}

__global__ void fastreg::device::interpolate_gradient_kernel(fastreg::device::VolumeGradient nabla_phi_n_psi,
                                                             fastreg::device::VolumeGradient nabla_phi_n_psi_t,
                                                             fastreg::device::DeformationField psi) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi.dims.x - 1 || y > psi.dims.y - 1) {
        return;
    }

    float3 idx   = make_float3(x, y, 0.f);
    float3 zstep = make_float3(0.f, 0.f, 1.f);

    int global_idx = y * nabla_phi_n_psi.dims.x + x;

    float4* nabla_phi_n_psi_t_ptr = nabla_phi_n_psi_t.beg(x, y);
    for (int i = 0; i <= psi.dims.z - 1; nabla_phi_n_psi_t_ptr = nabla_phi_n_psi_t.zstep(nabla_phi_n_psi_t_ptr),
             global_idx += nabla_phi_n_psi.dims.x * nabla_phi_n_psi.dims.y, idx += zstep, ++i) {
        float4 psi_val         = psi.data[global_idx];
        *nabla_phi_n_psi_t_ptr = interpolate_field(nabla_phi_n_psi, trunc(psi_val));
    }
}

void fastreg::device::interpolate_gradient(fastreg::device::VolumeGradient& nabla_phi_n_psi,
                                           fastreg::device::VolumeGradient& nabla_phi_n_psi_t,
                                           fastreg::device::DeformationField& psi) {
    dim3 block(64, 16);
    dim3 grid(divUp(psi.dims.x, block.x), divUp(psi.dims.y, block.y));

    interpolate_gradient_kernel<<<grid, block>>>(nabla_phi_n_psi, nabla_phi_n_psi_t, psi);
    cudaSafeCall(cudaGetLastError());
}

/*
 * LAPLACIAN
 */

__global__ void fastreg::device::interpolate_laplacian_kernel(fastreg::device::Laplacian L,
                                                              fastreg::device::Laplacian L_o_psi,
                                                              fastreg::device::DeformationField psi) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi.dims.x - 1 || y > psi.dims.y - 1) {
        return;
    }

    float4* psi_ptr     = psi.beg(x, y);
    float4* L_o_psi_ptr = L_o_psi.beg(x, y);
    for (int i = 0; i <= psi.dims.z - 1; psi_ptr = psi.zstep(psi_ptr), L_o_psi_ptr = L_o_psi.zstep(L_o_psi_ptr), ++i) {
        float4 psi_val = *psi_ptr;
        *L_o_psi_ptr   = interpolate_field(L, trunc(psi_val));
    }
}

void fastreg::device::interpolate_laplacian(fastreg::device::Laplacian& L, fastreg::device::Laplacian& L_o_psi,
                                            fastreg::device::DeformationField& psi) {
    dim3 block(64, 16);
    dim3 grid(divUp(psi.dims.x, block.x), divUp(psi.dims.y, block.y));

    interpolate_laplacian_kernel<<<grid, block>>>(L, L_o_psi, psi);
    cudaSafeCall(cudaGetLastError());
}

/*
 * SECOND ORDER DIFFERENTIATOR METHODS
 */

void fastreg::device::SecondOrderDifferentiator::calculate(fastreg::device::Laplacian& L) {
    dim3 block(64, 16);
    dim3 grid(divUp(L.dims.x, block.x), divUp(L.dims.y, block.y));

    estimate_laplacian_kernel<<<grid, block>>>(*this, L);
    cudaSafeCall(cudaGetLastError());
}

__global__ void fastreg::device::estimate_laplacian_kernel(const fastreg::device::SecondOrderDifferentiator diff,
                                                           fastreg::device::Laplacian L) {
    diff.laplacian(L);
}

__device__ void fastreg::device::SecondOrderDifferentiator::laplacian(fastreg::device::Laplacian& L) const {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x == 0 || x >= L.dims.x - 1 || y == 0 || y >= L.dims.y - 1) {
        return;
    }

    int idx_x_1 = x + 1;
    int idx_x_2 = x - 1;

    int idx_y_1 = y + 1;
    int idx_y_2 = y - 1;

    float4* L_ptr = L.beg(x, y) + L.dims.x * L.dims.y;

#pragma unroll
    for (int i = 1; i <= L.dims.z - 2; L_ptr = L.zstep(L_ptr), ++i) {
        int idx_z_1 = i + 1;
        int idx_z_2 = i - 1;

        float4 L_val = -6.f * *psi(x, y, i) + *psi(idx_x_1, y, i) + *psi(idx_x_2, y, i) + *psi(x, idx_y_1, i) +
                       *psi(x, idx_y_2, i) + *psi(x, y, idx_z_1) + *psi(x, y, idx_z_2);
        *L_ptr = -1.f * L_val;
    }
}

__global__ void fastreg::device::estimate_laplacian_velocity_kernel(
    const fastreg::device::SecondOrderDifferentiator diff, fastreg::device::Laplacian L) {
    diff.laplacian_velocity(L);
}

__device__ void fastreg::device::SecondOrderDifferentiator::laplacian_velocity(fastreg::device::Laplacian& L) const {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x <= 1 || x >= L.dims.x - 2 || y <= 1 || y >= L.dims.y - 1) {
        return;
    }

    int idx_x_1 = x + 2;
    int idx_x_2 = x + 1;
    int idx_x_3 = x - 1;
    int idx_x_4 = x - 2;

    int idx_y_1 = y + 2;
    int idx_y_2 = y - 1;
    int idx_y_3 = y - 1;
    int idx_y_4 = y - 2;

    float4* L_ptr = L.beg(x, y) + L.dims.x * L.dims.y * 2;

#pragma unroll
    for (int i = 2; i <= L.dims.z - 3; L_ptr = L.zstep(L_ptr), ++i) {
        int idx_z_1 = i + 2;
        int idx_z_2 = i - 1;
        int idx_z_3 = i - 1;
        int idx_z_4 = i - 2;

        float4 vel = *psi(x, y, i);

        float4 vel_xx = (vel.x < 0.f) ? *psi(idx_x_1, y, i) - 2.f * *psi(idx_x_2, y, i) + vel
                                      : vel - 2.f * *psi(idx_x_3, y, i) + *psi(idx_x_4, y, i);
        float4 vel_yy = (vel.y < 0.f) ? *psi(x, idx_y_1, i) - 2.f * *psi(x, idx_y_2, i) + vel
                                      : vel - 2.f * *psi(x, idx_y_3, i) + *psi(x, idx_y_4, i);
        float4 vel_zz = (vel.z < 0.f) ? *psi(x, y, idx_z_1) - 2.f * *psi(x, y, idx_z_2) + vel
                                      : vel - 2.f * *psi(x, y, idx_z_3) + *psi(x, y, idx_z_4);

        float4 L_val = vel_xx + vel_yy + vel_zz;
        *L_ptr       = -1.f * L_val;
    }
}

/*
 * (Dv)v
 */

__global__ void fastreg::device::calc_dv_v_kernel(fastreg::device::Velocity v, fastreg::device::DVV dv_v) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x == 0 || x >= v.dims.x - 1 || y == 0 || y >= v.dims.y - 1) {
        return;
    }

    int idx_x_1 = x + 1;
    int idx_x_2 = x - 1;

    int idx_y_1 = y + 1;
    int idx_y_2 = y - 1;

    float4* dv_v_ptr = dv_v.beg(x, y) + v.dims.x * v.dims.y;

#pragma unroll
    for (int i = 1; i <= dv_v.dims.z - 2; dv_v_ptr = dv_v.zstep(dv_v_ptr), ++i) {
        int idx_z_1 = i + 1;
        int idx_z_2 = i - 1;

        float4 vel = *v(x, y, i);

        /* entropy scheme */
        float aux_x_1 = fmaxf(vel.x, 0.f) * fmaxf(vel.x, 0.f) - fminf(vel.x, 0.f) * fminf(vel.x, 0.f) +
                        fminf((*v(idx_x_1, y, i)).x, 0.f) * fminf((*v(idx_x_1, y, i)).x, 0.f) -
                        fmaxf((*v(idx_x_1, y, i)).x, 0.f) * fmaxf((*v(idx_x_1, y, i)).x, 0.f);
        /* upwind scheme */
        float aux_x_2 =
            (vel.y > 0.f) ? vel.y * (vel.x - (*v(x, idx_y_2, i)).x) : vel.y * ((*v(x, idx_y_1, i)).x - vel.x);
        float aux_x_3 =
            (vel.z > 0.f) ? vel.z * (vel.x - (*v(x, y, idx_z_2)).x) : vel.z * ((*v(x, y, idx_z_1)).x - vel.x);

        float aux_y_1 =
            (vel.x > 0.f) ? vel.x * (vel.y - (*v(idx_x_2, y, i)).y) : vel.x * ((*v(idx_x_1, y, i)).y - vel.y);
        float aux_y_2 = fmaxf(vel.y, 0.f) * fmaxf(vel.y, 0.f) - fminf(vel.y, 0.f) * fminf(vel.y, 0.f) +
                        fminf((*v(x, idx_y_1, i)).y, 0.f) * fminf((*v(x, idx_y_1, i)).y, 0.f) -
                        fmaxf((*v(x, idx_y_1, i)).y, 0.f) * fmaxf((*v(x, idx_y_1, i)).y, 0.f);
        float aux_y_3 =
            ((*v(x, y, i)).z > 0.f) ? vel.z * (vel.y - (*v(x, y, idx_z_2)).y) : vel.z * ((*v(x, y, idx_z_1)).y - vel.y);

        float aux_z_1 =
            ((*v(x, y, i)).x > 0.f) ? vel.x * (vel.z - (*v(idx_x_2, y, i)).z) : vel.x * ((*v(idx_x_1, y, i)).z - vel.z);
        float aux_z_2 =
            ((*v(x, y, i)).y > 0.f) ? vel.y * (vel.z - (*v(x, idx_y_2, i)).z) : vel.y * ((*v(x, idx_y_1, i)).z - vel.z);
        float aux_z_3 = fmaxf(vel.z, 0.f) * fmaxf(vel.z, 0.f) - fminf(vel.z, 0.f) * fminf(vel.z, 0.f) +
                        fminf((*v(x, y, idx_z_1)).z, 0.f) * fminf((*v(x, y, idx_z_1)).z, 0.f) -
                        fmaxf((*v(x, y, idx_z_1)).z, 0.f) * fmaxf((*v(x, y, idx_z_1)).z, 0.f);

        *dv_v_ptr = make_float4(0.5f * aux_x_1 + aux_x_2 + aux_x_3, aux_y_1 + 0.5f * aux_y_2 + aux_y_3,
                                aux_z_1 + aux_z_2 + 0.5f * aux_z_3, 0.f);
    }
}

void fastreg::device::calc_dv_v(fastreg::device::Velocity& v, fastreg::device::DVV& dv_v) {
    dim3 block(64, 16);
    dim3 grid(divUp(v.dims.x, block.x), divUp(v.dims.y, block.y));

    calc_dv_v_kernel<<<grid, block>>>(v, dv_v);
    cudaSafeCall(cudaGetLastError());
}

/*
 * JACOBIAN
 */

__device__ __forceinline__ Mat4f* fastreg::device::Jacobian::beg(int x, int y) const { return data + x + dims.x * y; }

__device__ __forceinline__ Mat4f* fastreg::device::Jacobian::zstep(Mat4f* const ptr) const {
    return ptr + dims.x * dims.y;
}

__device__ __forceinline__ Mat4f* fastreg::device::Jacobian::operator()(int x, int y, int z) const {
    return data + x + y * dims.x + z * dims.y * dims.x;
}

void fastreg::device::clear(Jacobian& J) {
    dim3 block(64, 16);
    dim3 grid(divUp(J.dims.x, block.x), divUp(J.dims.y, block.y));

    clear_jacobian_kernel<<<grid, block>>>(J);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void fastreg::device::clear_jacobian_kernel(fastreg::device::Jacobian J) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > J.dims.x - 1 || y > J.dims.y - 1) {
        return;
    }

    Mat4f* beg = J.beg(x, y);
    Mat4f* end = beg + J.dims.x * J.dims.y * J.dims.z;

    for (Mat4f* pos = beg; pos != end; pos = J.zstep(pos)) {
        float4 g = make_float4(0.f, 0.f, 0.f, 0.f);

        Mat4f val;
        val.data[0] = g;
        val.data[1] = g;
        val.data[2] = g;

        *pos = val;
    }
}

/*
 * DIFFERENTIATOR METHODS
 */

__global__ void fastreg::device::estimate_jacobian_kernel(const fastreg::device::Differentiator diff,
                                                          fastreg::device::Jacobian J) {
    diff.jacobian(J);
}

void fastreg::device::Differentiator::calculate_jacobian(fastreg::device::Jacobian& J) {
    dim3 block(64, 16);
    dim3 grid(divUp(J.dims.x, block.x), divUp(J.dims.y, block.y));

    estimate_jacobian_kernel<<<grid, block>>>(*this, J);
    cudaSafeCall(cudaGetLastError());
}

__global__ void fastreg::device::estimate_deformation_jacobian_kernel(const fastreg::device::Differentiator diff,
                                                                      fastreg::device::Jacobian J) {
    diff.deformation_jacobian(J);
}

void fastreg::device::Differentiator::calculate_deformation_jacobian(fastreg::device::Jacobian& J) {
    dim3 block(64, 16);
    dim3 grid(divUp(J.dims.x, block.x), divUp(J.dims.y, block.y));

    estimate_deformation_jacobian_kernel<<<grid, block>>>(*this, J);
    cudaSafeCall(cudaGetLastError());
}

__global__ void fastreg::device::estimate_velocity_jacobian_kernel(const fastreg::device::Differentiator diff,
                                                                   fastreg::device::Jacobian J) {
    diff.velocity_jacobian(J);
}

void fastreg::device::Differentiator::calculate_velocity_jacobian(fastreg::device::Jacobian& J) {
    dim3 block(64, 16);
    dim3 grid(divUp(J.dims.x, block.x), divUp(J.dims.y, block.y));

    estimate_velocity_jacobian_kernel<<<grid, block>>>(*this, J);
    cudaSafeCall(cudaGetLastError());
}

__device__ __forceinline__ void fastreg::device::Differentiator::jacobian(fastreg::device::Jacobian& J) const {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x == 0 || x >= psi.dims.x - 1 || y == 0 || y >= psi.dims.y - 1) {
        return;
    }

    int idx_x_1 = x + 1;
    int idx_x_2 = x - 1;

    int idx_y_1 = y + 1;
    int idx_y_2 = y - 1;

    Mat4f* J_ptr = J.beg(x, y) + psi.dims.x * psi.dims.y;

#pragma unroll
    for (int i = 1; i <= psi.dims.z - 2; J_ptr = J.zstep(J_ptr), ++i) {
        int idx_z_1 = i + 1;
        int idx_z_2 = i - 1;

        float4 J_x = (*psi(idx_x_1, y, i) - *psi(idx_x_2, y, i)) / 2.f;
        float4 J_y = (*psi(x, idx_y_1, i) - *psi(x, idx_y_2, i)) / 2.f;
        float4 J_z = (*psi(x, y, idx_z_1) - *psi(x, y, idx_z_2)) / 2.f;

        Mat4f val;
        val.data[0] = make_float4(J_x.x, J_y.x, J_z.x, 0.f);
        val.data[1] = make_float4(J_x.y, J_y.y, J_z.y, 0.f);
        val.data[2] = make_float4(J_x.z, J_y.z, J_z.z, 0.f);

        *J(x, y, i) = val;
    }
}

__device__ __forceinline__ void fastreg::device::Differentiator::deformation_jacobian(
    fastreg::device::Jacobian& J) const {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi.dims.x - 1 || y > psi.dims.y - 1) {
        return;
    }

    int idx_x_1 = x + 1;
    int idx_x_2 = x - 1;
    if (x == 0) {
        idx_x_2 = x + 1;
    } else if (x == psi.dims.x - 1) {
        idx_x_1 = x - 1;
    }

    int idx_y_1 = y + 1;
    int idx_y_2 = y - 1;
    if (y == 0) {
        idx_y_2 = y + 1;
    } else if (y == psi.dims.y - 1) {
        idx_y_1 = y - 1;
    }

    Mat4f* J_ptr = J.beg(x, y);

#pragma unroll
    for (int i = 0; i <= psi.dims.z - 1; J_ptr = J.zstep(J_ptr), ++i) {
        int idx_z_1 = i + 1;
        int idx_z_2 = i - 1;
        if (i == 0) {
            idx_z_2 = i + 1;
        } else if (i == psi.dims.z - 1) {
            idx_z_1 = i - 1;
        }

        float4 J_x = (psi.get_displacement(idx_x_1, y, i) - psi.get_displacement(idx_x_2, y, i)) / 2.f;
        float4 J_y = (psi.get_displacement(x, idx_y_1, i) - psi.get_displacement(x, idx_y_2, i)) / 2.f;
        float4 J_z = (psi.get_displacement(x, y, idx_z_1) - psi.get_displacement(x, y, idx_z_2)) / 2.f;

        Mat4f val;
        val.data[0] = make_float4(J_x.x, J_y.x, J_z.x, 0.f);
        val.data[1] = make_float4(J_x.y, J_y.y, J_z.y, 0.f);
        val.data[2] = make_float4(J_x.z, J_y.z, J_z.z, 0.f);

        *J(x, y, i) = val;
    }
}

__device__ __forceinline__ void fastreg::device::Differentiator::velocity_jacobian(fastreg::device::Jacobian& J) const {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi.dims.x - 1 || y > psi.dims.y - 1) {
        return;
    }

    int idx_x_1 = x + 1;
    int idx_x_2 = x - 1;
    if (x == 0) {
        idx_x_2 = x;
    } else if (x == psi.dims.x - 1) {
        idx_x_1 = x;
    }

    int idx_y_1 = y + 1;
    int idx_y_2 = y - 1;
    if (y == 0) {
        idx_y_2 = y;
    } else if (y == psi.dims.y - 1) {
        idx_y_1 = y;
    }

    Mat4f* J_ptr = J.beg(x, y);

#pragma unroll
    for (int i = 0; i <= psi.dims.z - 1; J_ptr = J.zstep(J_ptr), ++i) {
        int idx_z_1 = i + 1;
        int idx_z_2 = i - 1;
        if (i == 0) {
            idx_z_2 = i;
        } else if (i == psi.dims.z - 1) {
            idx_z_1 = i;
        }

        float4 vel = *psi(x, y, i);

        float4 J_x = (vel.x < 0.f) ? *psi(idx_x_1, y, i) - vel : vel - *psi(idx_x_2, y, i);
        float4 J_y = (vel.y < 0.f) ? *psi(x, idx_y_1, i) - vel : vel - *psi(x, idx_y_2, i);
        float4 J_z = (vel.z < 0.f) ? *psi(x, y, idx_z_1) - vel : vel - *psi(x, y, idx_z_2);

        Mat4f val;
        val.data[0] = make_float4(J_x.x, J_y.x, J_z.x, 0.f);
        val.data[1] = make_float4(J_x.y, J_y.y, J_z.y, 0.f);
        val.data[2] = make_float4(J_x.z, J_y.z, J_z.z, 0.f);

        *J(x, y, i) = val;
    }
}
