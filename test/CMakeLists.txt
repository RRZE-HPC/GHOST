find_package(GHOST REQUIRED PATHS ${CMAKE_BINARY_DIR} NO_DEFAULT_PATH)

include_directories(${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing ${GHOST_CXX_FLAGS}")

add_executable(simple simple.cpp)

add_dependencies(simple ghost)
target_link_libraries(simple ${HWLOC_LIBRARIES} ${MPI_LIBRARIES} ${SCOTCH_LIBRARIES} ${CUDA_LIBRARIES} ${LIKWID_LIBRARIES} ${COLPACK_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

if (${BUILD_SHARED_LIBS})
    target_link_libraries(simple ${CMAKE_BINARY_DIR}/libghost.so)
else ()
    target_link_libraries(simple ${CMAKE_BINARY_DIR}/libghost.a)
endif()
