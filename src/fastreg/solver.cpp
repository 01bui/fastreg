#include <solver.hpp>

/*
 * SOLVER
 */

fastreg::cuda::Solver::Solver(Params& params) {
    /*
     * PARAMETERS
     */

    /* volume */
    cv::Vec3i d = params.volume_dims;
    dims        = make_int3(d[0], d[1], d[2]);
    no_voxels   = dims.x * dims.y * dims.z;

    /* solver */
    solver_params.max_iter        = params.max_iter;
    solver_params.verbosity       = params.verbosity;
    solver_params.max_update_norm = params.max_update_norm;
    solver_params.no_labels       = params.no_labels;

    solver_params.rho_0      = params.rho_0;
    solver_params.reset_iter = params.reset_iter;
    solver_params.tau        = params.tau;

    solver_params.alpha = params.alpha;
    solver_params.w_reg = params.w_reg;

    solver_params.p = params.p;
    solver_params.C = params.C;

    /*
     * SOLVER HELPER CLASSES
     */

    /* density and velocity field--used in the kinetic term */
    v   = new fastreg::cuda::Velocity(d);
    rho = new fastreg::cuda::Density(d, solver_params.rho_0);
    div = new fastreg::cuda::Divergence(d);

    v_device   = new fastreg::device::Velocity(v->get_data().ptr<float4>(), dims);
    rho_device = new fastreg::device::Density(rho->get_data().ptr<float>(), dims);
    div_device = new fastreg::device::Divergence(div->get_data().ptr<float>(), dims);

    /* gradients */
    spatial_grads = new fastreg::cuda::SpatialGradients(d);

    nabla_phi_n = new fastreg::device::VolumeGradient(spatial_grads->nabla_phi_n->get_data().ptr<float4>(), dims);
    nabla_phi_n_o_psi =
        new fastreg::device::VolumeGradient(spatial_grads->nabla_phi_n_o_psi->get_data().ptr<float4>(), dims);
    J           = new fastreg::device::Jacobian(spatial_grads->J->get_data().ptr<Mat4f>(), dims);
    J_inv       = new fastreg::device::Jacobian(spatial_grads->J_inv->get_data().ptr<Mat4f>(), dims);
    L           = new fastreg::device::Laplacian(spatial_grads->L->get_data().ptr<float4>(), dims);
    L_o_psi_inv = new fastreg::device::Laplacian(spatial_grads->L_o_psi_inv->get_data().ptr<float4>(), dims);
    dv_v        = new fastreg::device::DVV(spatial_grads->dv_v->get_data().ptr<float4>(), dims);
    delta_v     = new fastreg::device::Laplacian(spatial_grads->delta_v->get_data().ptr<float4>(), dims);
    nabla_U     = new fastreg::device::PotentialGradient(spatial_grads->nabla_U->get_data().ptr<float4>(), dims);

    spatial_grads_device = new fastreg::device::SpatialGradients(nabla_phi_n, nabla_phi_n_o_psi, J, J_inv, L,
                                                                 L_o_psi_inv, dv_v, delta_v, nabla_U);

    /* reductor */
    r = new fastreg::device::Reductor(dims);
}

fastreg::cuda::Solver::~Solver() = default;

void fastreg::cuda::Solver::clear() {
    v->clear();
    rho->clear(solver_params.rho_0);
    div->clear();

    spatial_grads->nabla_phi_n->clear();
    spatial_grads->nabla_phi_n_o_psi->clear();
    spatial_grads->J->clear();
    spatial_grads->J_inv->clear();
    spatial_grads->L->clear();
    spatial_grads->L_o_psi_inv->clear();
    spatial_grads->dv_v->clear();
    spatial_grads->delta_v->clear();
    spatial_grads->nabla_U->clear();
}

void fastreg::cuda::Solver::estimate_psi(const std::shared_ptr<fastreg::cuda::Volume> phi_global,
                                         std::shared_ptr<fastreg::cuda::Volume> phi_global_psi_inv,
                                         const std::shared_ptr<fastreg::cuda::Volume> phi_n,
                                         std::shared_ptr<fastreg::cuda::Volume> phi_n_psi,
                                         std::shared_ptr<fastreg::cuda::DeformationField> psi,
                                         std::shared_ptr<fastreg::cuda::DeformationField> psi_inv) {
    /* DEVICE CLASSES */
    fastreg::device::DeformationField psi_device(psi->get_data().ptr<float4>(), dims);
    fastreg::device::DeformationField psi_inv_device(psi_inv->get_data().ptr<float4>(), dims);

    fastreg::device::Volume phi_global_device(phi_global->get_data().ptr<float2>(), dims);
    fastreg::device::Volume phi_global_psi_inv_device(phi_global_psi_inv->get_data().ptr<float2>(), dims);

    fastreg::device::Volume phi_n_device(phi_n->get_data().ptr<float2>(), dims);
    fastreg::device::Volume phi_n_psi_device(phi_n_psi->get_data().ptr<float2>(), dims);

    SDFs sdfs(phi_global_device, phi_global_psi_inv_device, phi_n_device, phi_n_psi_device);

    fastreg::device::VolumeDifferentiator vol_diff(phi_n_device);
    fastreg::device::Differentiator diff(psi_device);
    fastreg::device::Differentiator diff_inv(psi_inv_device);
    fastreg::device::Differentiator diff_v(*v_device);
    fastreg::device::SecondOrderDifferentiator second_order_diff(psi_device);
    fastreg::device::SecondOrderDifferentiator velocity_second_order_diff(*v_device);

    Differentiators differentiators(vol_diff, diff, diff_inv, diff_v, second_order_diff, velocity_second_order_diff);

    /* run the solver */
    fastreg::device::estimate_psi(sdfs, psi_device, psi_inv_device, *v_device, *rho_device, *div_device,
                                  spatial_grads_device, differentiators, r, solver_params);
}
