find_package(GHOST REQUIRED CONFIG PATHS ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_BINARY_DIR})

#set(CMAKE_C_LINK_EXECUTABLE "icc")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GHOST_C_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GHOST_CXX_FLAGS}")
#set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${GHOST_Fortran_FLAGS}")

link_libraries(${HWLOC_LIBRARIES} ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${CMAKE_BINARY_DIR}/libghost_cuda.a ${CMAKE_BINARY_DIR}/libghost.so ${SCOTCH_LIBRARIES} ${CUDA_LIBRARIES})
if (USE_CUDA)
    add_executable(simple simple.c)
endif()
