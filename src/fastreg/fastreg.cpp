#include <fastreg.hpp>

static void get_image_data(std::shared_ptr<fastreg::cuda::Volume> img, float *data, short *labels_data,
                           Params &params) {
    /* copy data to host */
    float2 *data_with_labels_host = new float2[params.no_voxels()];
    img->get_data().download(data_with_labels_host);

    /* get image intensities */
    float *data_host        = new float[params.no_voxels()];
    short *data_labels_host = new short[params.no_voxels()];

    for (int i = 0; i < params.no_voxels(); i++) {
        data_host[i]        = data_with_labels_host[i].x;
        data_labels_host[i] = (short) data_with_labels_host[i].y;
    }

    /* copy data to device */
    cudaSafeCall(cudaMemcpy(data, data_host, params.no_voxels() * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(labels_data, data_labels_host, params.no_voxels() * sizeof(short), cudaMemcpyHostToDevice));
    delete data_host, data_labels_host, data_with_labels_host;
}

FastReg::FastReg(const Params &params_) : frame_counter_(0), params(params_) {
    /*
     * INITIALISATION OF PHI_GLOBAL, PHI_GLOBAL(PSI_INV), PHI_N, AND PHI_N(PSI)
     */

    phi_global                = std::make_shared<fastreg::cuda::Volume>(params.volume_dims);
    phi_global_warped_to_live = std::make_shared<fastreg::cuda::Volume>(params.volume_dims);
    phi_n                     = std::make_shared<fastreg::cuda::Volume>(params.volume_dims);
    phi_n_psi                 = std::make_shared<fastreg::cuda::Volume>(params.volume_dims);

    /*
     * INITIALISATION OF PSI AND PSI_INV
     */

    psi     = std::make_shared<fastreg::cuda::DeformationField>(params.volume_dims);
    psi_inv = std::make_shared<fastreg::cuda::DeformationField>(params.volume_dims);

    /*
     * INITIALISATION OF THE SOLVER
     */

    solver = std::make_shared<fastreg::cuda::Solver>(params);
}

FastReg::~FastReg() = default;

Params &FastReg::getParams() { return params; }

std::shared_ptr<fastreg::cuda::DeformationField> FastReg::getDeformationField() { return psi; }

/*
 * PIPELINE
 */

bool FastReg::operator()(float *image_data_1, short *labels_data_1, float *image_data_2, short *labels_data_2) {
    phi_global->initialise(image_data_1, labels_data_1);
    phi_global_warped_to_live->initialise(image_data_1, labels_data_1);

    phi_n->initialise(image_data_2, labels_data_2);
    phi_n_psi->initialise(image_data_2, labels_data_2);

    /*
     * ESTIMATION OF DEFORMATION FIELD AND SURFACE FUSION
     */

    solver->estimate_psi(phi_global, phi_global_warped_to_live, phi_n, phi_n_psi, psi, psi_inv);

    get_image_data(phi_global_warped_to_live, image_data_1, labels_data_1, params);
    get_image_data(phi_n_psi, image_data_2, labels_data_2, params);

    return ++frame_counter_, true;
}
