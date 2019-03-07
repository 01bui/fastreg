/* fastreg includes */
#include <cuda/scalar_fields.hpp>

/*
 * SCALAR FIELD
 */

fastreg::cuda::ScalarField::ScalarField(cv::Vec3i dims_) {
    dims = make_int3(dims_[0], dims_[1], dims_[2]);

    int no_voxels = dims.x * dims.y * dims.z;
    data.create(no_voxels * sizeof(float));
    clear();
}

fastreg::cuda::ScalarField::~ScalarField() = default;

CudaData fastreg::cuda::ScalarField::get_data() { return data; }

const CudaData fastreg::cuda::ScalarField::get_data() const { return data; }

int3 fastreg::cuda::ScalarField::get_dims() { return dims; }

void fastreg::cuda::ScalarField::clear() {
    fastreg::device::ScalarField field(data.ptr<float>(), dims);
    fastreg::device::clear(field);
}

float fastreg::cuda::ScalarField::sum() {
    fastreg::device::ScalarField field(data.ptr<float>(), dims);

    float result = fastreg::device::reduce_sum(field);
    return result;
}

void fastreg::cuda::ScalarField::print() {
    int sizes[3] = {dims.x, dims.y, dims.z};

    cv::Mat* mat = new cv::Mat(3, sizes, CV_32FC1);
    data.download(mat->ptr<float>());

    std::cout << "--- FIELD ---" << std::endl;
    for (int i = 0; i < dims.x; i++) {
        for (int j = 0; j < dims.y; j++) {
            for (int k = 0; k < dims.z; k++) {
                float val = mat->at<float>(k, j, i);

                std::cout << "(x,y,z)=(" << i << ", " << j << ", " << k << "), val=(" << val << std::endl;
            }
        }
    }

    delete mat;
}

/*
 * DENSITY
 */

fastreg::cuda::Density::Density(cv::Vec3i dims_, float rho_0_) : ScalarField(dims_) { clear(rho_0_); }

fastreg::cuda::Density::~Density() = default;

void fastreg::cuda::Density::clear(float rho_0) {
    fastreg::device::Density rho(get_data().ptr<float>(), get_dims());
    fastreg::device::init_density(rho, rho_0);
}
