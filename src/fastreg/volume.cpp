/* fastreg includes */
#include <cuda/volume.hpp>

/*
 * VOLUME
 */

fastreg::cuda::Volume::Volume(cv::Vec3i dims_) : dims(dims_) {
    int no_voxels = dims[0] * dims[1] * dims[2];
    data.create(no_voxels * sizeof(float2));
    clear();
}

fastreg::cuda::Volume::~Volume() = default;

cv::Vec3i fastreg::cuda::Volume::get_dims() const { return dims; }

CudaData fastreg::cuda::Volume::get_data() { return data; }

const CudaData fastreg::cuda::Volume::get_data() const { return data; }

void fastreg::cuda::Volume::set_data(CudaData& data) { data = data; }

void fastreg::cuda::Volume::initialise(float* image_data, short* labels_data) {
    int3 d = device_cast<int3>(dims);

    fastreg::device::Volume vol(data.ptr<float2>(), d);
    fastreg::device::initialise(vol, image_data, labels_data);
}

void fastreg::cuda::Volume::clear() {
    int3 d = device_cast<int3>(dims);

    fastreg::device::Volume field(data.ptr<float2>(), d);
    fastreg::device::clear(field);
}

void fastreg::cuda::Volume::print() {
    int sizes[3] = {dims[0], dims[1], dims[2]};

    cv::Mat* mat = new cv::Mat(3, sizes, CV_32FC2);
    get_data().download(mat->ptr<float2>());

    std::cout << "--- IMAGE ---" << std::endl;
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                float intensity_val = mat->at<float2>(k, j, i).x;
                float label_val     = mat->at<float2>(k, j, i).y;

                std::cout << "(x,y,z)=(" << i << ", " << j << ", " << k << "), intensity: " << intensity_val
                          << ", label: " << label_val << std::endl;
            }
        }
    }

    delete mat;
}
