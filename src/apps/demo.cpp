/* fastreg includes */
#include <fastreg.hpp>

/* boost includes */
#include <boost/program_options.hpp>

/* itk includes */
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>

/* sys headers */
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

typedef itk::Image<short, 3> ImageTypeShort;
typedef itk::Image<float, 3> ImageTypeFloat;

typedef itk::ImageFileReader<ImageTypeShort> ImageTypeShortReader;
typedef itk::ImageFileReader<ImageTypeFloat> ImageTypeFloatReader;

/*
 * parse file name from a path
 */

std::string get_file_name(const std::string &s) {
    char sep = '/';

#ifdef _WIN32
    sep = '\\';
#endif

    size_t i = s.rfind(sep, s.length());
    if (i != std::string::npos) {
        return (s.substr(i + 1, s.length() - i));
    }

    return ("");
}

/*
 * app
 */

struct FastRegApp {
    FastRegApp(std::string params_path, std::string target_path, std::string target_labels_path,
               std::string source_path, std::string source_labels_path, std::string out_path, bool verbose,
               bool vverbose)
        : exit_(false),
          params_path_(params_path),
          image_name_1(target_path),
          labels_name_1(target_labels_path),
          image_name_2(source_path),
          labels_name_2(source_labels_path),
          out_path_(out_path),
          verbose_(verbose),
          vverbose_(vverbose) {
        /*
         * initialise parameters
         */

        if (verbose_) {
            params.verbosity = 1;
        } else if (vverbose_) {
            params.verbosity = 2;
        }

        /*
         * declare parameters to read from .ini
         */

        boost::program_options::options_description desc("parameters");
        declare_parameters(desc, params);

        /*
         * read parameters from .ini
         */

        boost::program_options::variables_map vm;
        load_parameters(desc, vm);

        /*
         * initialise sobolevfusion
         */

        fastreg = std::make_shared<FastReg>(params);
    }

    /*
     * declare fastreg parameters
     */

    void declare_parameters(boost::program_options::options_description &desc, Params &params) {
        /*
         * image
         */

        desc.add_options()("VOL_DIMS_X", boost::program_options::value<int>(&params.volume_dims[0]),
                           "no. of voxels along x axis");
        desc.add_options()("VOL_DIMS_Y", boost::program_options::value<int>(&params.volume_dims[1]),
                           "no. of voxels along y axis");
        desc.add_options()("VOL_DIMS_Z", boost::program_options::value<int>(&params.volume_dims[2]),
                           "no. of voxels along z axis");

        desc.add_options()("VOL_SIZE_X", boost::program_options::value<float>(&params.volume_size[0]),
                           "vol. size along x axis (metres)");
        desc.add_options()("VOL_SIZE_Y", boost::program_options::value<float>(&params.volume_size[1]),
                           "vol. size along y axis (metres)");
        desc.add_options()("VOL_SIZE_Z", boost::program_options::value<float>(&params.volume_size[2]),
                           "vol. size along z axis (metres)");

        desc.add_options()("NO_LABELS", boost::program_options::value<int>(&params.no_labels), "no. of segmentations");

        /*
         * solver
         */
        desc.add_options()("MAX_ITER", boost::program_options::value<int>(&params.max_iter),
                           "max. no. of iterations of the solver");
        desc.add_options()("MAX_UPDATE_NORM", boost::program_options::value<float>(&params.max_update_norm),
                           "max. update norm when running the solver");

        /* FASTFUSION */
        desc.add_options()("RHO_0", boost::program_options::value<float>(&params.rho_0), "initial density");

        desc.add_options()("RESET_ITER", boost::program_options::value<int>(&params.reset_iter),
                           "iteration when to reset t in order to prevent underdamping");
        desc.add_options()("TAU", boost::program_options::value<float>(&params.tau), "velocity diffusion coefficient");

        desc.add_options()("ALPHA", boost::program_options::value<float>(&params.alpha), "gradient descent step size");
        desc.add_options()("W_REG", boost::program_options::value<float>(&params.w_reg), "regularisation weight");
        desc.add_options()("p", boost::program_options::value<float>(&params.p),
                           "scaling factor for momentum and potential energy");
        desc.add_options()("C", boost::program_options::value<float>(&params.C),
                           "scaling factor for momentum and potential energy");
    }

    /*
     * read parameters from params.ini
     */

    void load_parameters(boost::program_options::options_description &desc, boost::program_options::variables_map &vm) {
        std::ifstream settings_file(params_path_);

        boost::program_options::store(boost::program_options::parse_config_file(settings_file, desc), vm);
        boost::program_options::notify(vm);
    }

    /*
     * load images
     */

    void load_images() {
        /* initialise file names */
        std::string image_file_name_1 = image_name_1;
        std::string image_file_name_2 = image_name_2;

        /* load images */
        ImageTypeFloatReader::Pointer reader_image_1 = ImageTypeFloatReader::New();
        reader_image_1->SetFileName(image_file_name_1.c_str());
        reader_image_1->Update();

        ImageTypeFloatReader::Pointer reader_image_2 = ImageTypeFloatReader::New();
        reader_image_2->SetFileName(image_file_name_2.c_str());
        reader_image_2->Update();

        /* get the images */
        image_1 = reader_image_1->GetOutput();
        image_2 = reader_image_2->GetOutput();
    }

    /*
     * load labels
     */

    void load_labels() {
        /* init file names */
        std::string labels_file_name_1 = labels_name_1;
        std::string labels_file_name_2 = labels_name_2;

        /* load labels */
        ImageTypeShortReader::Pointer reader_labels_1 = ImageTypeShortReader::New();
        reader_labels_1->SetFileName(labels_file_name_1.c_str());
        reader_labels_1->Update();

        ImageTypeShortReader::Pointer reader_labels_2 = ImageTypeShortReader::New();
        reader_labels_2->SetFileName(labels_file_name_2.c_str());
        reader_labels_2->Update();

        /* get the labels */
        image_labels_1 = reader_labels_1->GetOutput();
        image_labels_2 = reader_labels_2->GetOutput();
    }

    /*
     * save warped images
     */

    void save_warped_images() {
        /*
         * images
         */

        std::string input_1 = image_name_1;

        itk::ImageIOBase::Pointer imageIO =
            itk::ImageIOFactory::CreateImageIO(input_1.c_str(), itk::ImageIOFactory::ReadMode);

        imageIO->SetFileName(input_1);
        imageIO->ReadImageInformation();

        itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
        nifti_io->SetPixelType(imageIO->GetPixelType());

        itk::ImageFileWriter<ImageTypeFloat>::Pointer writer = itk::ImageFileWriter<ImageTypeFloat>::New();

        /* re-sampled image 1 */
        writer->SetFileName(out_path_ + "/target_psi_inv_" + get_file_name(image_name_2) + ".nii");
        writer->SetInput(image_1);
        writer->SetImageIO(nifti_io);
        writer->Update();
        /* re-sampled image 2 */
        writer->SetFileName(out_path_ + "/source_psi_" + get_file_name(image_name_2) + ".nii");
        writer->SetInput(image_2);
        writer->Update();

        /*
         * labels
         */

        std::string input_2 = labels_name_1;

        itk::ImageIOBase::Pointer imageIO_labels =
            itk::ImageIOFactory::CreateImageIO(input_2.c_str(), itk::ImageIOFactory::ReadMode);

        imageIO_labels->SetFileName(input_2);
        imageIO_labels->ReadImageInformation();

        nifti_io->SetPixelType(imageIO_labels->GetPixelType());
        itk::ImageFileWriter<ImageTypeShort>::Pointer writer_labels = itk::ImageFileWriter<ImageTypeShort>::New();

        /* labels 1 */
        writer_labels->SetFileName(out_path_ + "/target_psi_inv_labels_" + get_file_name(image_name_2) + ".nii");
        writer_labels->SetInput(image_labels_1);
        writer_labels->SetImageIO(nifti_io);
        writer_labels->Update();

        /* labels 2 */
        writer_labels->SetFileName(out_path_ + "/source_psi_labels_" + get_file_name(image_name_2) + ".nii");
        writer_labels->SetInput(image_labels_2);
        writer_labels->SetImageIO(nifti_io);
        writer_labels->Update();

        std::cout << "--- SAVED OUTPUT" << std::endl;
    }

    /*
     * execute
     */

    bool execute() {
        /* fastreg app */
        fastreg = fastreg;

        /* declare device pointers for images */
        float *image_data_1, *image_data_2;
        cudaSafeCall(cudaMalloc((void **) &image_data_1, params.no_voxels() * sizeof(float)));
        cudaSafeCall(cudaMalloc((void **) &image_data_2, params.no_voxels() * sizeof(float)));

        /* declare device pointers for labels */
        short *labels_data_1, *labels_data_2;
        cudaSafeCall(cudaMalloc((void **) &labels_data_1, params.no_voxels() * sizeof(short)));
        cudaSafeCall(cudaMalloc((void **) &labels_data_2, params.no_voxels() * sizeof(short)));

        std::cout << "-------" << std::endl;
        std::cout << "TARGET: " << image_name_1 << ", " << labels_name_1 << std::endl;
        std::cout << "SOURCE: " << image_name_2 << ", " << labels_name_2 << std::endl;

        /*
         * NON-RIGID REGISTRATION
         */

        std::cout << "-------" << std::endl;
        std::cout << "RUNNING NON-RIGID REGISTRATION" << std::endl;

        double time_ms = 0;
        bool has_image = false;

        /* load images and labels */
        load_images();
        load_labels();

        cudaSafeCall(cudaMemcpy(image_data_1, image_1->GetBufferPointer(), params.no_voxels() * sizeof(float),
                                cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(labels_data_1, image_labels_1->GetBufferPointer(), params.no_voxels() * sizeof(short),
                                cudaMemcpyHostToDevice));

        cudaSafeCall(cudaMemcpy(image_data_2, image_2->GetBufferPointer(), params.no_voxels() * sizeof(float),
                                cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(labels_data_2, image_labels_2->GetBufferPointer(), params.no_voxels() * sizeof(short),
                                cudaMemcpyHostToDevice));

        /* run registartion */
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); /* measure time */
        has_image = (*(fastreg))(image_data_1, labels_data_1, image_data_2, labels_data_2);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "--- REGISTRATION TIME: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

        /* copy warped data */
        cudaSafeCall(cudaMemcpy(image_1->GetBufferPointer(), image_data_1, params.no_voxels() * sizeof(float),
                                cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(image_2->GetBufferPointer(), image_data_2, params.no_voxels() * sizeof(float),
                                cudaMemcpyDeviceToHost));

        cudaSafeCall(cudaMemcpy(image_labels_1->GetBufferPointer(), labels_data_1, params.no_voxels() * sizeof(short),
                                cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(image_labels_2->GetBufferPointer(), labels_data_2, params.no_voxels() * sizeof(short),
                                cudaMemcpyDeviceToHost));

        save_warped_images();

        return true;
    }

    std::string image_name_1, labels_name_1, image_name_2, labels_name_2; /* file names */

    /* images and labels */
    ImageTypeFloat::Pointer image_1, image_2;
    ImageTypeShort::Pointer image_labels_1, image_labels_2;

    /* pipeline */
    Params params;
    std::shared_ptr<FastReg> fastreg;

    /* app parameters */
    bool exit_, logger_, verbose_, vverbose_;
    std::string params_path_, out_path_;
};

/*
 * parse the input flags and determine the file paths
 */

void parse_flags(std::vector<std::string> args, std::string *params_path, std::string *target_path,
                 std::string *target_labels_path, std::string *source_path, std::string *source_labels_path,
                 std::string *out_path, bool *verbose, bool *vverbose) {
    std::vector<std::string> flags = {"-h", "--help", "--verbose", "--vverbose"};

    int idx = 0;
    for (auto arg : args) {
        if (std::find(std::begin(flags), std::end(flags), arg) != std::end(flags)) {
            if (arg == "-h" || arg == "--help") {
                std::cout << "USAGE: fastreg [OPTIONS] <ini path> <target path> <target labels path> <source path> "
                             "<source labels path> <out path>"
                          << std::endl;
                std::cout << "\t--help -h:    display help" << std::endl;
                std::cout << "\t--verbose: low verbosity" << std::endl;
                std::cout << "\t--vverbose: high verbosity" << std::endl;
                std::exit(EXIT_SUCCESS);
            }

            if (arg == "--verbose") {
                *verbose = true;
            }
            if (arg == "--vverbose") {
                *vverbose = true;
            }
        } else if (idx == 0) {
            *params_path = arg;
            idx++;
        } else if (idx == 1) {
            *target_path = arg;
            idx++;
        } else if (idx == 2) {
            *target_labels_path = arg;
            idx++;
        } else if (idx == 3) {
            *source_path = arg;
            idx++;
        } else if (idx == 4) {
            *source_labels_path = arg;
            idx++;
        } else if (idx == 5) {
            *out_path = arg;
            idx++;
        }
    }
}

/*
 * main
 */

int main(int argc, char *argv[]) {
    int device = 0;

    /* print info. about the gpu */
    fastreg::cuda::setDevice(device);
    fastreg::cuda::printShortCudaDeviceInfo(device);

    /* program requires at least 5 arguments */
    if (argc < 6) {
        return std::cerr << "error: incorrect number of arguments; please supply path to source and target data and "
                            ".ini file; "
                            "exiting..."
                         << std::endl,
               -1;
    }

    std::vector<std::string> args(argv + 1, argv + argc);
    std::string params_path, target_path, target_labels_path, source_path, source_labels_path, out_path;

    bool verbose  = false;
    bool vverbose = false;

    parse_flags(args, &params_path, &target_path, &target_labels_path, &source_path, &source_labels_path, &out_path,
                &verbose, &vverbose);

    /* execute */
    FastRegApp app(params_path, target_path, target_labels_path, source_path, source_labels_path, out_path, verbose,
                   vverbose);
    app.execute();

    return 0;
}
