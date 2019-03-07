# Build all fastreg source files into library fastreg
add_module_library(fastreg)

# Link to fastreg library
target_link_libraries(fastreg
    gcov
    pthread
    Threads::Threads
    ${Boost_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${LAPACK_LIBRARIES}
    ${PCL_LIBRARIES}
)
