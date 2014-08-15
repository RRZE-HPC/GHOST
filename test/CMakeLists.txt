find_package(GHOST REQUIRED CONFIG PATHS ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_BINARY_DIR})

add_executable(simple simple.cpp)

add_dependencies(simple ghost)
target_link_libraries(simple ${HWLOC_LIBRARIES} ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${CMAKE_BINARY_DIR}/libghost.so ${SCOTCH_LIBRARIES} ${CUDA_LIBRARIES} ${LIKWID_LIBRARIES} ${COLPACK_LIBRARIES})
