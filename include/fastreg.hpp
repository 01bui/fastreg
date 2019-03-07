#pragma once

/* fastreg includes */
#include <core.hpp>
#include <params.hpp>
#include <solver.hpp>

/* sys headers */
#include <math.h>
#include <thread>

class FastReg {
public:
    /* default constructor */
    FastReg(const Params &params);
    /* default destructor */
    ~FastReg();

    /* get fastreg params */
    Params &getParams();
    /* get the deformation field */
    std::shared_ptr<fastreg::cuda::DeformationField> getDeformationField();

    bool operator()(float *image_data_1, short *labels_data_1, float *image_data_2, short *labels_data_2);

private:
    /* frame counter */
    int frame_counter_;
    /* parameters */
    Params params;

    /* volumes */
    std::shared_ptr<fastreg::cuda::Volume> phi_global, phi_global_warped_to_live, phi_n, phi_n_psi;
    /* deformation fields */
    std::shared_ptr<fastreg::cuda::DeformationField> psi, psi_inv;
    /* solver */
    std::shared_ptr<fastreg::cuda::Solver> solver;
};

/* get image intensities w/o labels */
static void get_image_data(std::shared_ptr<fastreg::cuda::Volume> vol, float *data, Params &params);
