find_package(GHOST REQUIRED PATHS ${CMAKE_BINARY_DIR} NO_DEFAULT_PATH)

include_directories(${CMAKE_BINARY_DIR})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Fujitsu")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -SSL2BLAMP")
endif()

add_executable(simple simple.cpp)
add_executable(minimal minimal.c)
add_executable(context context.c)

set(TESTS simple minimal context)

foreach(TEST ${TESTS})
    add_dependencies(${TEST} ghost)
    add_dependencies(${TEST} ghost)

    if (${BUILD_SHARED_LIBS})
        target_link_libraries(${TEST} ${CMAKE_BINARY_DIR}/libghost.so)
    else ()
        target_link_libraries(${TEST} ${CMAKE_BINARY_DIR}/libghost.a)
    endif()

    target_link_libraries(${TEST} ${HWLOC_LIBRARIES} ${MPI_LIBRARIES} ${SCOTCH_LIBRARIES} ${CUDA_LIBRARIES} ${LIKWID_LIBRARIES} ${COLPACK_LIBRARIES} ${ZOLTAN_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} -lpthread)
endforeach()
