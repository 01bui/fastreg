#pragma once

/* fastreg includes */
#include <cuda/device_memory.hpp>
#include <cuda/volume.hpp>

#include <internal.hpp>
#include <params.hpp>
#include <precomp.hpp>

/* sys headers */
#include <memory>

namespace fastreg {
namespace cuda {

/*
 * VECTOR FIELD
 */

class VectorField {
public:
    /* constructor */
    VectorField(cv::Vec3i dims_);
    /* destructor */
    ~VectorField();

    /* get dimensions of the field */
    cv::Vec3i get_dims() const;

    /* get the data in the field */
    CudaData get_data();
    const CudaData get_data() const;
    /* set the data in the field */
    void set_data(CudaData &data);

    /* clear field */
    void clear();
    /* print field */
    void print();
    /* get no. of nan's in the field */
    int get_no_nans();

protected:
    CudaData data;  /* field data */
    cv::Vec3i dims; /* field dimensions */
};

/*
 * DEFORMATION FIELD
 */

class DeformationField : public VectorField {
public:
    /* constructor */
    DeformationField(cv::Vec3i dims_);
    /* destructor */
    ~DeformationField();

    /* init field to identity */
    void clear();

    /* apply the field to an sdf */
    void apply(std::shared_ptr<fastreg::cuda::Volume> phi, std::shared_ptr<fastreg::cuda::Volume> phi_psi);
    /* approximate the inverse of the field */
    void get_inverse(fastreg::cuda::DeformationField &psi_inv);
};

/*
 * VOLUME GRADIENT, FIELD LAPLACIAN, POTENTIAL GRADIENT, VELOCITY, DVV
 */

typedef VectorField VolumeGradient;
typedef VectorField DivergencePartials;
typedef VectorField Laplacian;
typedef VectorField PotentialGradient;
typedef VectorField Velocity;
typedef VectorField DVV;

/*
 * JACOBIAN
 */

class Jacobian {
public:
    /* constructor */
    Jacobian(cv::Vec3i dims_);
    /* destructor */
    ~Jacobian();

    CudaData get_data();
    const CudaData get_data() const;

    /* clear jacobian */
    void clear();

private:
    CudaData data;
    cv::Vec3i dims;
};

typedef Jacobian VolumeHessian;

/*
 * SPATIAL GRADIENTS
 */

struct SpatialGradients {
    /* constructor */
    SpatialGradients(cv::Vec3i dims_);
    /* destructor */
    ~SpatialGradients();

    VolumeGradient *nabla_phi_n, *nabla_phi_n_o_psi;
    Jacobian *J, *J_inv;
    Laplacian *L, *L_o_psi_inv;
    DVV *dv_v;
    DivergencePartials *nabla_div;
    Laplacian *delta_v;
    PotentialGradient *nabla_U;
};

}  // namespace cuda
}  // namespace fastreg

namespace fastreg {
namespace device {

/*
 * VECTOR FIELD
 */

struct VectorField {
    /* constructor */
    VectorField(float4 *const data_, const int3 dims_) : data(data_), dims(dims_) {}

    __device__ __forceinline__ float4 *beg(int x, int y) const;
    __device__ __forceinline__ float4 *zstep(float4 *const ptr) const;
    __device__ __forceinline__ float4 *operator()(int x, int y, int z) const;

    __device__ __forceinline__ float4 get_displacement(int x, int y, int z) const;

    float4 *const data; /* vector field data */
    const int3 dims;    /* vector field dimensions */
};

/* clear field */
__global__ void clear_kernel(VectorField field);
void clear(VectorField &field);

/*
 * DEFORMATION FIELD
 */

typedef VectorField DeformationField;

/* set deformation field to identity */
__global__ void init_identity_kernel(DeformationField field);
void init_identity(DeformationField &field);
/* apply psi to an sdf */
__global__ void apply_kernel(const Volume phi, Volume phi_warped, const DeformationField psi);
void apply(const Volume &phi, Volume &phi_warped, const fastreg::device::DeformationField &psi);
/* approximate the inverse of a deformation field */
__global__ void estimate_inverse_kernel(fastreg::device::DeformationField psi,
                                        fastreg::device::DeformationField psi_inv);
void estimate_inverse(fastreg::device::DeformationField &psi, fastreg::device::DeformationField &psi_inv);

/*
 * DIVERGENCE
 */

typedef VectorField DivergencePartials;

/*
 * LAPLACIAN
 */

typedef VectorField Laplacian;

struct SecondOrderDifferentiator {
    /* constructor */
    SecondOrderDifferentiator(fastreg::device::DeformationField &psi_) : psi(psi_) {}

    __device__ void divergence_partials(fastreg::device::DivergencePartials &nabla_div) const;
    void calculate_divergence_partials(fastreg::device::DivergencePartials &nabla_div);

    __device__ void laplacian(fastreg::device::Laplacian &L) const;
    __device__ void laplacian_velocity(fastreg::device::Laplacian &L) const;
    void calculate(fastreg::device::Laplacian &L);

    fastreg::device::DeformationField psi;
};

/* esitmate divergence */
__global__ void estimate_divergence_partials_kernel(const fastreg::device::SecondOrderDifferentiator diff,
                                                    fastreg::device::DivergencePartials nabla_div);
/* estimate laplacian */
__global__ void estimate_laplacian_kernel(const fastreg::device::SecondOrderDifferentiator diff,
                                          fastreg::device::Laplacian L);
__global__ void estimate_laplacian_velocity_kernel(const fastreg::device::SecondOrderDifferentiator diff,
                                                   fastreg::device::Laplacian L);

/* interpolate laplacian */
__global__ void interpolate_laplacian_kernel(fastreg::device::Laplacian L, fastreg::device::Laplacian L_o_psi,
                                             fastreg::device::DeformationField psi);
void interpolate_laplacian(fastreg::device::Laplacian &L, fastreg::device::Laplacian &L_o_psi,
                           fastreg::device::DeformationField &psi);

/*
 * POTENTIAL GRADIENT
 */

typedef VectorField PotentialGradient;

/*
 * VELOCITY
 */

typedef VectorField Velocity;
typedef VectorField DVV;

__global__ void calc_dv_v_kernel(Velocity v, DVV dv_v);
void calc_dv_v(Velocity &v, DVV &dv_v);

/*
 * JACOBIAN
 */

struct Jacobian {
    /* constructor */
    Jacobian(Mat4f *const data_, int3 dims_) : data(data_), dims(dims_) {}

    __device__ __forceinline__ Mat4f *beg(int x, int y) const;
    __device__ __forceinline__ Mat4f *zstep(Mat4f *const ptr) const;
    __device__ __forceinline__ Mat4f *operator()(int x, int y, int z) const;

    Mat4f *const data; /* jacobian data */
    const int3 dims;
};

typedef Jacobian VolumeHessian;

struct Differentiator {
    /* constructor */
    Differentiator(fastreg::device::VectorField &psi_) : psi(psi_) {}

    /* calculate jacobian */
    __device__ void jacobian(fastreg::device::Jacobian &J) const;
    void calculate_jacobian(fastreg::device::Jacobian &J);

    __device__ void deformation_jacobian(fastreg::device::Jacobian &J) const;
    void calculate_deformation_jacobian(fastreg::device::Jacobian &J);

    __device__ void velocity_jacobian(fastreg::device::Jacobian &J) const;
    void calculate_velocity_jacobian(fastreg::device::Jacobian &J);

    fastreg::device::VectorField psi;
};

/* clear jacobian */
__global__ void clear_jacobian_kernel(Jacobian J);
void clear(Jacobian &J);

/* estimate jacobian */
__global__ void estimate_jacobian_kernel(const fastreg::device::Differentiator diff, fastreg::device::Jacobian J);
__global__ void estimate_deformation_jacobian_kernel(const fastreg::device::Differentiator diff,
                                                     fastreg::device::Jacobian J);
__global__ void estimate_velocity_jacobian_kernel(const fastreg::device::Differentiator diff,
                                                  fastreg::device::Jacobian J);

/*
 * VOLUME GRADIENT
 */

typedef VectorField VolumeGradient;

struct VolumeDifferentiator {
    /* constructor */
    VolumeDifferentiator(Volume &vol_) : vol(vol_) {}

    /* calculate the gradient */
    __device__ void operator()(fastreg::device::VolumeGradient &grad) const;
    void calculate(fastreg::device::VolumeGradient &grad);
    /* calculate the hessian */
    __device__ void operator()(fastreg::device::VolumeHessian &H) const;
    void calculate(fastreg::device::VolumeHessian &H);

    Volume vol;
};

/* estimate volume gradient */
__global__ void estimate_gradient_kernel(const fastreg::device::VolumeDifferentiator diff,
                                         fastreg::device::VolumeGradient grad);
/* estimate volume hessian */
__global__ void estimate_hessian_kernel(const fastreg::device::VolumeDifferentiator diff,
                                        fastreg::device::VolumeHessian H);

/* interpolate volume gradient */
__global__ void interpolate_gradient_kernel(fastreg::device::VolumeGradient nabla_phi_n_psi,
                                            fastreg::device::VolumeGradient nabla_phi_n_psi_t,
                                            fastreg::device::DeformationField psi);
void interpolate_gradient(fastreg::device::VolumeGradient &nabla_phi_n_psi,
                          fastreg::device::VolumeGradient &nabla_phi_n_psi_t, fastreg::device::DeformationField &psi);

/*
 * SPATIAL GRADIENTS
 */

struct SpatialGradients {
    SpatialGradients(fastreg::device::VolumeGradient *nabla_phi_n_, fastreg::device::VolumeGradient *nabla_phi_n_o_psi_,
                     fastreg::device::Jacobian *J_, fastreg::device::Jacobian *J_inv_, fastreg::device::Laplacian *L_,
                     fastreg::device::Laplacian *L_o_psi_inv_, fastreg::device::DVV *dv_v_,
                     fastreg::device::Laplacian *delta_v_, fastreg::device::PotentialGradient *nabla_U_)
        : nabla_phi_n(nabla_phi_n_),
          nabla_phi_n_o_psi(nabla_phi_n_o_psi_),
          J(J_),
          J_inv(J_inv_),
          L(L_),
          L_o_psi_inv(L_o_psi_inv_),
          dv_v(dv_v_),
          delta_v(delta_v_),
          nabla_U(nabla_U_) {}
    ~SpatialGradients();

    fastreg::device::VolumeGradient *nabla_phi_n, *nabla_phi_n_o_psi;
    fastreg::device::Jacobian *J, *J_inv;
    fastreg::device::Laplacian *L, *L_o_psi_inv;
    fastreg::device::DVV *dv_v;
    fastreg::device::Laplacian *delta_v;
    fastreg::device::PotentialGradient *nabla_U;
};

}  // namespace device
}  // namespace fastreg
