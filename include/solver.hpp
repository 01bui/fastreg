#pragma once

/* fastreg includes */
#include <cuda/reductor.hpp>
#include <cuda/scalar_fields.hpp>
#include <cuda/vector_fields.hpp>
#include <cuda/volume.hpp>

#include <safe_call.hpp>
#include <types.hpp>

/* sys headers */
#include <memory>

/*
 * SOLVER PARAMETERS
 */

struct SolverParams {
    int verbosity, max_iter, reset_iter, no_labels;
    float max_update_norm, tau, rho_0, alpha, w_reg, p, C;
};

/*
 * SDF'S USED IN THE SOLVER
 */

struct SDFs {
    SDFs(fastreg::device::Volume &phi_global_, fastreg::device::Volume &phi_global_psi_inv_,
         fastreg::device::Volume &phi_n_, fastreg::device::Volume &phi_n_psi_)
        : phi_global(phi_global_), phi_global_psi_inv(phi_global_psi_inv_), phi_n(phi_n_), phi_n_psi(phi_n_psi_) {}

    fastreg::device::Volume phi_global, phi_global_psi_inv, phi_n, phi_n_psi;
};

/*
 * SPATIAL DIFFERENTIATORS
 */

struct Differentiators {
    Differentiators(fastreg::device::VolumeDifferentiator &volume_diff_, fastreg::device::Differentiator &diff_,
                    fastreg::device::Differentiator &diff_inv_, fastreg::device::Differentiator &diff_v_,
                    fastreg::device::SecondOrderDifferentiator &second_order_diff_,
                    fastreg::device::SecondOrderDifferentiator &velocity_second_order_diff_)
        : volume_diff(volume_diff_),
          diff(diff_),
          diff_inv(diff_inv_),
          diff_v(diff_v_),
          second_order_diff(second_order_diff_),
          velocity_second_order_diff(velocity_second_order_diff_) {}

    fastreg::device::VolumeDifferentiator volume_diff;
    fastreg::device::Differentiator diff, diff_inv, diff_v;
    fastreg::device::SecondOrderDifferentiator second_order_diff, velocity_second_order_diff;
};

namespace fastreg {
namespace cuda {

/*
 * SOLVER
 */

class Solver {
public:
    /* constructor */
    Solver(Params &params);
    /* destructor */
    ~Solver();

    /* clear all spatial derivatives */
    void clear();
    /* estimate deformation */
    void estimate_psi(std::shared_ptr<fastreg::cuda::Volume> phi_global,
                      std::shared_ptr<fastreg::cuda::Volume> phi_global_psi_inv,
                      std::shared_ptr<fastreg::cuda::Volume> phi_n, std::shared_ptr<fastreg::cuda::Volume> phi_n_psi,
                      std::shared_ptr<fastreg::cuda::DeformationField> psi,
                      std::shared_ptr<fastreg::cuda::DeformationField> psi_inv);

private:
    /* volume params */
    int3 dims;
    float3 voxel_sizes;

    int no_voxels;
    float trunc_dist, eta, max_weight;

    /* solver params */
    SolverParams solver_params;

    /* density and velocity fields--used in the kinetic term */
    fastreg::cuda::Velocity *v;
    fastreg::device::Velocity *v_device;

    fastreg::cuda::Density *rho;
    fastreg::device::Density *rho_device;

    fastreg::cuda::Divergence *div;
    fastreg::device::Divergence *div_device;

    /* gradients */
    fastreg::cuda::SpatialGradients *spatial_grads;
    fastreg::device::SpatialGradients *spatial_grads_device;

    fastreg::device::VolumeGradient *nabla_phi_n, *nabla_phi_n_o_psi;
    fastreg::device::Jacobian *J, *J_inv;
    fastreg::device::Laplacian *L, *L_o_psi_inv, *delta_v;
    fastreg::device::DVV *dv_v;
    fastreg::device::PotentialGradient *nabla_U;

    /* used to calculate value of the energy functional */
    fastreg::device::Reductor *r;
};
}  // namespace cuda

namespace device {

/* potential gradient */
__global__ void calculate_potential_gradient_kernel(float2 *phi_n, float2 *phi_global_o_psi_inv, float4 *nabla_phi_n,
                                                    Mat4f *J_inv, float4 *L_o_psi_inv, float4 *nabla_U, float w_reg,
                                                    int dim_x, int dim_y, int dim_z);
void calculate_potential_gradient(fastreg::device::Volume &phi_n, fastreg::device::Volume &phi_global_o_psi_inv,
                                  fastreg::device::VolumeGradient &nabla_phi_n, fastreg::device::Jacobian &J_inv,
                                  fastreg::device::Laplacian &L_o_psi_inv, fastreg::device::PotentialGradient &nabla_U,
                                  float w_reg);

/* divergence */
__global__ void calc_divergence_kernel(float *rho, float4 *v, float *div, int dim_x, int dim_y, int dim_z);
void calc_divergence(fastreg::device::Density &rho, fastreg::device::Velocity &v, fastreg::device::Divergence &div);

/* estimate psi */
void estimate_psi(SDFs &sdfs, fastreg::device::DeformationField &psi, fastreg::device::DeformationField &psi_inv,
                  fastreg::device::Velocity &v, fastreg::device::Density &rho, fastreg::device::Divergence &div,
                  fastreg::device::SpatialGradients *spatial_grads, Differentiators &diffs,
                  fastreg::device::Reductor *r, SolverParams &params);

/*
 * PDE'S
 */

/* DENSITY PDE */
__global__ void update_density_kernel(float *rho, float *div, float alpha, int dim_x, int dim_y, int dim_z);
void update_density(fastreg::device::Density &rho, fastreg::device::Divergence &div, float alpha);

/* VELOCITY PDE */
__global__ void update_velocity_kernel(float4 *v, float4 *dv_v, float4 *delta_v, float *rho, float4 *nabla_U,
                                       float alpha, float t, float tau, float p, float C, int dim_x, int dim_y,
                                       int dim_z);
void update_velocity(fastreg::device::Velocity &v, fastreg::device::DVV &dv_v, fastreg::device::Laplacian &delta_v,
                     fastreg::device::Density &rho, fastreg::device::DeformationField &nabla_U, float alpha, float t,
                     float tau, float p, float C);

/* DEFORMATION FIELD PDE */
__global__ void update_psi_kernel(float4 *psi, float4 *v, float4 *updates, float alpha, int dim_x, int dim_y,
                                  int dim_z);
void update_psi(fastreg::device::DeformationField &psi, fastreg::device::Velocity &v, float4 *updates, float alpha);

/* INVERSE DEFORMATION FIELD PDE */
__global__ void update_psi_inv_kernel(float4 *psi_inv, Mat4f *J_psi_inv, float4 *v, float alpha, int dim_x, int dim_y,
                                      int dim_z);
void update_psi_inv(fastreg::device::DeformationField &psi_inv, fastreg::device::Jacobian &J_psi_inv,
                    fastreg::device::Velocity &v, float alpha);

/*
 * DICE
 */

static void calc_dice_coefficient(fastreg::device::Volume &image_1, fastreg::device::Volume &image_2, int no_labels);

}  // namespace device
}  // namespace fastreg
