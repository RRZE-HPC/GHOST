find_package(GHOST REQUIRED PATHS ${CMAKE_BINARY_DIR} NO_DEFAULT_PATH)

include_directories(${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing ${GHOST_CXX_FLAGS}")

add_executable(simple simple.cpp)

add_dependencies(simple ghost)
target_link_libraries(simple ${HWLOC_LIBRARIES} ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${CMAKE_BINARY_DIR}/libghost.so ${SCOTCH_LIBRARIES} ${CUDA_LIBRARIES} ${LIKWID_LIBRARIES} ${COLPACK_LIBRARIES})
