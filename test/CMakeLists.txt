find_package(GHOST REQUIRED CONFIG PATHS ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_BINARY_DIR})

link_libraries(${HWLOC_LIBRARIES} ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${CMAKE_BINARY_DIR}/libghost.so ${SCOTCH_LIBRARIES} ${CUDA_LIBRARIES} ${LIKWID_LIBRARIES})
if (USE_CUDA)
    add_executable(simple simple.c)
endif()
