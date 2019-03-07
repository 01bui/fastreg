/* fastreg includes */
#include <cuda/vector_fields.hpp>

/*
 * VECTOR FIELD
 */

fastreg::cuda::VectorField::VectorField(cv::Vec3i dims_) : dims(dims_) {
    int no_voxels = dims[0] * dims[1] * dims[2];
    data.create(no_voxels * sizeof(float4));
    clear();
}

fastreg::cuda::VectorField::~VectorField() = default;

cv::Vec3i fastreg::cuda::VectorField::get_dims() const { return dims; }

CudaData fastreg::cuda::VectorField::get_data() { return data; }

const CudaData fastreg::cuda::VectorField::get_data() const { return data; }

void fastreg::cuda::VectorField::set_data(CudaData& data) { data = data; }

void fastreg::cuda::VectorField::clear() {
    int3 d = make_int3(dims[0], dims[1], dims[2]);

    fastreg::device::VectorField field(data.ptr<float4>(), d);
    fastreg::device::clear(field);
}

void fastreg::cuda::VectorField::print() {
    int sizes[3] = {dims[0], dims[1], dims[2]};

    cv::Mat* mat = new cv::Mat(3, sizes, CV_32FC4);
    get_data().download(mat->ptr<float4>());

    std::cout << "--- FIELD ---" << std::endl;
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                float u = mat->at<float4>(k, j, i).x;
                float v = mat->at<float4>(k, j, i).y;
                float w = mat->at<float4>(k, j, i).z;

                if (fabs(u) > 1e-5f || fabs(v) > 1e-5f || fabs(w) > 1e-5f) {
                    std::cout << "(x,y,z)=(" << i << ", " << j << ", " << k << "), (u,v,w)=(" << u << ", " << v << ","
                              << w << ")" << std::endl;
                }
            }
        }
    }

    delete mat;
}

int fastreg::cuda::VectorField::get_no_nans() {
    int sizes[3] = {dims[0], dims[1], dims[2]};

    cv::Mat* mat = new cv::Mat(3, sizes, CV_32FC4);
    get_data().download(mat->ptr<float4>());

    int no_nan = 0;
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                float u = mat->at<float4>(k, j, i).x;
                float v = mat->at<float4>(k, j, i).y;
                float w = mat->at<float4>(k, j, i).z;

                if (std::isnan(u) || std::isnan(v) || std::isnan(w)) {
                    no_nan++;
                }
            }
        }
    }

    delete mat;
    return no_nan;
}

/*
 * DEFORMATION FIELD
 */

fastreg::cuda::DeformationField::DeformationField(cv::Vec3i dims_) : VectorField(dims_) { clear(); }

fastreg::cuda::DeformationField::~DeformationField() = default;

void fastreg::cuda::DeformationField::clear() {
    int3 d = fastreg::device_cast<int3>(get_dims());

    fastreg::device::DeformationField psi(get_data().ptr<float4>(), d);
    fastreg::device::init_identity(psi);
}

void fastreg::cuda::DeformationField::get_inverse(fastreg::cuda::DeformationField& psi_inv) {
    int3 d = fastreg::device_cast<int3>(get_dims());

    fastreg::device::DeformationField psi_device(get_data().ptr<float4>(), d);
    fastreg::device::DeformationField psi_inverse_device(psi_inv.get_data().ptr<float4>(), d);

    fastreg::device::estimate_inverse(psi_device, psi_inverse_device);
}

void fastreg::cuda::DeformationField::apply(std::shared_ptr<fastreg::cuda::Volume> phi_n,
                                            std::shared_ptr<fastreg::cuda::Volume> phi_n_psi) {
    int3 d = fastreg::device_cast<int3>(phi_n->get_dims());

    fastreg::device::Volume phi_device(phi_n->get_data().ptr<float2>(), d);
    fastreg::device::Volume phi_warped_device(phi_n_psi->get_data().ptr<float2>(), d);
    fastreg::device::DeformationField psi_device(get_data().ptr<float4>(), d);

    fastreg::device::apply(phi_device, phi_warped_device, psi_device);
}

/*
 * JACOBIAN
 */

fastreg::cuda::Jacobian::Jacobian(cv::Vec3i dims_) : dims(dims_) {
    int no_voxels = dims[0] * dims[1] * dims[2];
    data.create(no_voxels * sizeof(Mat4f));
    clear();
}

fastreg::cuda::Jacobian::~Jacobian() = default;

CudaData fastreg::cuda::Jacobian::get_data() { return data; }

const CudaData fastreg::cuda::Jacobian::get_data() const { return data; }

void fastreg::cuda::Jacobian::clear() {
    int3 d = make_int3(dims[0], dims[1], dims[2]);

    fastreg::device::Jacobian J(data.ptr<Mat4f>(), d);
    fastreg::device::clear(J);
}

/*
 * SPATIAL GRADIENTS
 */

fastreg::cuda::SpatialGradients::SpatialGradients(cv::Vec3i dims_) {
    nabla_phi_n       = new fastreg::cuda::VolumeGradient(dims_);
    nabla_phi_n_o_psi = new fastreg::cuda::VolumeGradient(dims_);
    J                 = new fastreg::cuda::Jacobian(dims_);
    J_inv             = new fastreg::cuda::Jacobian(dims_);
    L                 = new fastreg::cuda::Laplacian(dims_);
    L_o_psi_inv       = new fastreg::cuda::Laplacian(dims_);
    dv_v              = new fastreg::cuda::DVV(dims_);
    delta_v           = new fastreg::cuda::Laplacian(dims_);
    nabla_U           = new fastreg::cuda::PotentialGradient(dims_);
}

fastreg::cuda::SpatialGradients::~SpatialGradients() {
    delete nabla_phi_n, nabla_phi_n_o_psi, J, J_inv, L, L_o_psi_inv, dv_v, delta_v, nabla_U;
}
