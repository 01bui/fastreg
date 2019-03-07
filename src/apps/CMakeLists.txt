# Link source files to executables
add_application(app)

# enable all the io modules to support in itk
include(${ITK_USE_FILE})

# Link to all libraries
target_link_libraries(app
    fastreg
    png
    Boost::program_options
    ${Boost_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${ITK_LIBRARIES}
    ${OpenCV_LIBS}
)
