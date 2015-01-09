cmake_minimum_required (VERSION 2.8)
project (GHOST)

set(GHOST_VERSION_MAJOR 0)
set(GHOST_VERSION_MINOR 8)
set(GHOST_VERSION_REVISION 0)
set(GHOST_VERSION_IDENTIFIER "alpha")
set(GHOST_VERSION "${GHOST_VERSION_MAJOR}.${GHOST_VERSION_MINOR}.${GHOST_VERSION_REVISION}")
if (DEFINED GHOST_VERSION_IDENTIFIER)
    set(GHOST_VERSION "${GHOST_VERSION}-${GHOST_VERSION_IDENTIFIER}")
endif()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

include(CheckCCompilerFlag)
include(${PROJECT_SOURCE_DIR}/cmake/OptimizeForArchitecture.cmake)

option(USE_MPI "Build with MPI" 1)
option(USE_OPENMP "Build with OpenMP" 1)
option(USE_CUDA "Build with CUDA" 0)
option(USE_SCOTCH "Use PT-Scotch for matrix re-ordering" 0)
option(USE_COLPACK "Use ColPack for matrix coloring" 0)
option(USE_LONGIDX_LOCAL "Enable 64-bit indices for local quantities" 0)
option(USE_LONGIDX_GLOBAL "Enable 64-bit indices for global quantities" 1)
option(BUILD_MIC "Build for Intel MIC" 0)
option(ERR_WARN_INFO_PRINTONCE "Print all errors, warnings and info only once for each occurence" 1)
option(TRACK_DATATRANSFERS "Track MPI and CUDA data transfers" 0)
option(GATHER_GLOBAL_INFO "Gather some global information (should be disabled for large-scale runs)" 0)
option(GHOST_TSMTTSM_KAHAN "Use Kahan summation in the hand-vectorized TSMTTSM kernel" 0)
set(CFG_BLOCKVECTOR_SIZES "1" CACHE STRING "The block vector sizes for which generated kernels are being created (comma-separated list)")
set(CFG_SELL_CHUNKHEIGHTS "32" CACHE STRING "The SELL chunk heights for which generated kernels are being created (comma-separated list)")
set(CFG_SELL_CUDA_KERNELS "true,true,true,true,true,true;true,false,false,false,false,false;false,false,false,false,false,false" CACHE STRING "The permutations of SELL SpMV CUDA kernels to instantiate (semicolon-separated list of comma-separated lists). AXP(B)Y,SCALE,(V)SHIFT,DOT_YY,DOT_XY,DOT_XX")
set(LIBTYPE "SHARED" CACHE STRING "Type of GHOST build (required). Valid values: SHARED, STATIC")
set(VERBOSITY 1 CACHE STRING "Level of verbosity information. Valid values: 0 (be totally silent), 1 (print warnings, errors and further important information), 2, 3, 4 (increasing level of verbosity)")
set(INSTRUMENTATION "" CACHE STRING "Type of instrumentation (semicolon separated list). Valid values: TIMING, LIKWID")
set(INSTRUMENTATION_FUNCTYPES "(functype | GHOST_FUNCTYPE_MATH) && (functype | GHOST_FUNCTYPE_KERNEL)" CACHE STRING "Logical selection of function types to instrument. Check the variable functype against valid values as defined in func_util.h. Example: functype & (GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_COMMUNICATION) ")

find_package(PkgConfig)

add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE)
if (USE_LONGIDX_LOCAL)
    set(GHOST_HAVE_LONGIDX_LOCAL 1)
    if (NOT USE_LONGIDX_GLOBAL)
        message(FATAL_ERROR "Having 64-bit indices locally and only 32-bit indices globally is not allowed (and not sensible)!")
    endif()
endif()
if (USE_LONGIDX_GLOBAL)
    set(GHOST_HAVE_LONGIDX_GLOBAL 1)
    set(GHOST_HAVE_LONGIDX 1) # backward compatibility
endif()

if (TRACK_DATATRANSFERS)
    set(GHOST_HAVE_TRACK_DATATRANSFERS 1)
endif()

if (GATHER_GLOBAL_INFO)
    add_definitions(-DGHOST_GATHER_GLOBAL_INFO)
endif()

set(GHOST_VERBOSITY ${VERBOSITY})
set(GHOST_ERR_WARN_INFO_PRINTONCE ${ERR_WARN_INFO_PRINTONCE})

set (SOURCES
    src/bincrs.cpp
    src/crs.cpp
    src/crs_kacz.cpp
    src/crs_spmv.cpp
    src/datatransfers.cpp 
    src/densemat_cm_averagehalo.cpp
    src/densemat_cm.cpp 
    src/densemat_rm_averagehalo.cpp
    src/densemat_rm.cpp 
    src/sell.cpp
    src/sell_kacz.cpp
    src/sell_spmv.cpp
    src/sparsemat.cpp
    src/sparsemat_perm_colpack.cpp
    src/svqb.cpp
    src/timing.cpp
    src/tsmm_kernelmap.cpp
    src/tsmm_inplace_kernelmap.cpp
    src/tsmttsm_kernelmap.cpp
    src/bincrs.c
    src/blas_util.c
    src/carp.c
    src/context.c
    src/core.c
    src/crs.c 
    src/cu_util.c
    src/densemat.c
    src/densemat_cm.c 
    src/densemat_rm.c 
    src/error.c
    src/gemm.c
    src/instr.c
    src/locality.c
    src/machine.c
    src/math.c
    src/matrixmarket.c
    src/mmio.c
    src/omp.c
    src/pumap.c
    src/rand.c
    src/sell.c 
    src/sparsemat.c
    src/sparsemat_perm_scotch.c
    src/spmv_overlap.c 
    src/spmv_nompi.c 
    src/spmv_task.c
    src/spmv_vector.c 
    src/task.c
    src/taskq.c
    src/thpool.c
    src/timing.c
    src/tsmm.c
    src/tsmm_inplace.c
    src/tsmttsm.c
    src/types.c
    src/util.c 
    )

set (HEADERS
    include/ghost/bincrs.h
    include/ghost/bindensemat.h
    include/ghost/bitmap.h
    include/ghost/blas_mangle.h
    include/ghost/carp.h
    include/ghost/complex.h
    include/ghost/context.h
    include/ghost/core.h
    include/ghost/crs.h
    include/ghost/cu_util.h
    include/ghost/densemat.h
    include/ghost/densemat_cm.h
    include/ghost/densemat_iter_macros.h
    include/ghost/densemat_rm.h
    include/ghost/error.h
    include/ghost/helper.h
    include/ghost/instr.h
    include/ghost/locality.h	
    include/ghost/log.h
    include/ghost/machine.h
    include/ghost/math.h
    include/ghost/matrixmarket.h
    include/ghost/omp.h
    include/ghost/perm.h
    include/ghost/pumap.h
    include/ghost/rand.h
    include/ghost/sell.h
    include/ghost/sparsemat.h
    include/ghost/spmv.h
    include/ghost/spmv_solvers.h
    include/ghost/task.h
    include/ghost/taskq.h
    include/ghost/thpool.h
    include/ghost/timing.h
    include/ghost/tsmm.h
    include/ghost/tsmm_inplace.h
    include/ghost/tsmttsm.h
    include/ghost/types.h
    include/ghost/util.h
    )

string(REPLACE "," ";" CFG_SELL_CHUNKHEIGHTS_LIST ${CFG_SELL_CHUNKHEIGHTS})
string(REPLACE "," ";" CFG_BLOCKVECTOR_SIZES_LIST ${CFG_BLOCKVECTOR_SIZES})

set(SELL_CU_SPMV_SWITCHCASCADE "bool do_fallback = false;\\\nswitch \(SELL(mat)->chunkHeight) {\\\n")
foreach (SELL_CHUNKHEIGHT ${CFG_SELL_CHUNKHEIGHTS_LIST})
    set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}    case ${SELL_CHUNKHEIGHT}:\\\n")
    set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}        switch \(rhs->traits.ncols) {\\\n")
    foreach (SELL_NCOLS ${CFG_BLOCKVECTOR_SIZES_LIST})
        set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}            case ${SELL_NCOLS}:\\\n")
        foreach(SELL_CUDA_VARIANT ${CFG_SELL_CUDA_KERNELS})
            set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}                if(!(strcmp(boolstr,\"${SELL_CUDA_VARIANT}\"))) { kernel<dt1,dt2,dt2_b,${SELL_CHUNKHEIGHT},${SELL_NCOLS},${SELL_CUDA_VARIANT}><<<grid,block,smem>>>((dt2 *)lhsval,*(int *)(lhs->stride),(dt2 *)rhsval,*(int *)rhs->stride,flags,mat->nrows,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,(dt2 *)cu_shift,(dt2)scale,(dt2)beta,(dt2 *)cu_localdot); } else \\\n")
        endforeach()
        set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}                { PERFWARNING_LOG(\"Non-configured bool variant: %s\",boolstr);\\\n                do_fallback = true; }\\\n")
        set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}        break;\\\n")
    endforeach() 
    set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}            default:\\\n                PERFWARNING_LOG(\"Non-configured ncols %d!\",rhs->traits.ncols);\\\n                do_fallback = true;\\\n        }\\\n        break;\\\n")
endforeach()
set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}    default:\\\n        PERFWARNING_LOG(\"Non-configured chunk height: %d!\",SELL(mat)->chunkHeight);\\\n        do_fallback = true;\\\n}\\\n")

configure_file(
    ${PROJECT_SOURCE_DIR}/src/sell_spmv.cu.in
    ${PROJECT_BINARY_DIR}/src/sell_spmv.cu
    )

configure_file (
    ${PROJECT_SOURCE_DIR}/src/sell_spmv_sse.c.in
    ${PROJECT_BINARY_DIR}/src/sell_spmv_sse.c
    )
configure_file (
    ${PROJECT_SOURCE_DIR}/src/sell_spmv_avx.c.in
    ${PROJECT_BINARY_DIR}/src/sell_spmv_avx.c
    )
configure_file (
    ${PROJECT_SOURCE_DIR}/src/sell_spmv_mic.c.in
    ${PROJECT_BINARY_DIR}/src/sell_spmv_mic.c
    )
configure_file (
    ${PROJECT_SOURCE_DIR}/src/tsmm.c.in
    ${PROJECT_BINARY_DIR}/src/tsmm.c
    )
configure_file (
    ${PROJECT_SOURCE_DIR}/src/tsmm_inplace.c.in
    ${PROJECT_BINARY_DIR}/src/tsmm_inplace.c
    )
configure_file (
    ${PROJECT_SOURCE_DIR}/src/tsmttsm.c.in
    ${PROJECT_BINARY_DIR}/src/tsmttsm.c
    )
configure_file (
    ${PROJECT_SOURCE_DIR}/src/tsmttsm_avx.c.in
    ${PROJECT_BINARY_DIR}/src/tsmttsm_avx.c
    )

set (TO_PREPROCESS
    ${PROJECT_BINARY_DIR}/src/sell_spmv_avx.c
    ${PROJECT_BINARY_DIR}/src/sell_spmv_sse.c
    ${PROJECT_BINARY_DIR}/src/sell_spmv_mic.c
    ${PROJECT_BINARY_DIR}/src/tsmm.c
    ${PROJECT_BINARY_DIR}/src/tsmm_inplace.c
    ${PROJECT_BINARY_DIR}/src/tsmttsm.c
    ${PROJECT_BINARY_DIR}/src/tsmttsm_avx.c
    )

set (PP_SCRIPT ${PROJECT_SOURCE_DIR}/bin/ghost_pp.pl)
set (PP_HEADER_SCRIPT ${PROJECT_SOURCE_DIR}/bin/ghost_extractfunc.pl)
set (PP_KERNELMAP_SCRIPT ${PROJECT_SOURCE_DIR}/bin/ghost_mapfunc.pl)
set (PP_SRC_DIR ${PROJECT_BINARY_DIR}/src_pp/)
set (PP_INC_DIR ${PROJECT_BINARY_DIR}/ghost/)

foreach (TO_PP_FILE ${TO_PREPROCESS})
    get_filename_component(TO_PP_FILENAME ${TO_PP_FILE} NAME)
    get_filename_component(TO_PP_FILENAME_WE ${TO_PP_FILE} NAME_WE)
    set(PP_FILE ${PP_SRC_DIR}/${TO_PP_FILENAME})
    set(PP_HEADER ${PP_INC_DIR}/${TO_PP_FILENAME_WE}_gen.h)
    set(PP_KERNELMAP ${PP_INC_DIR}/${TO_PP_FILENAME_WE}.def)
    add_custom_command(
        OUTPUT ${PP_FILE}
        DEPENDS ${TO_PP_FILE} ${PP_SCRIPT} 
        COMMAND mkdir -p ${PP_SRC_DIR} && perl ${PP_SCRIPT} ${TO_PP_FILE} > ${PP_FILE}
        )
    add_custom_command(
        OUTPUT ${PP_HEADER}
        DEPENDS ${PP_FILE} ${PP_HEADER_SCRIPT} 
        COMMAND mkdir -p ${PP_INC_DIR} && perl ${PP_HEADER_SCRIPT} ${PP_FILE} > ${PP_HEADER}
        )
    add_custom_command(
        OUTPUT ${PP_KERNELMAP}
        DEPENDS ${PP_HEADER} ${PP_KERNELMAP_SCRIPT} 
        COMMAND mkdir -p ${PP_INC_DIR} && perl ${PP_KERNELMAP_SCRIPT} ${PP_HEADER} > ${PP_KERNELMAP}
        )
    list(APPEND PP_FILES ${PP_FILE} ${PP_HEADER} ${PP_KERNELMAP})
    list(APPEND HEADERS ${PP_HEADER})
endforeach()

set (SOURCES ${SOURCES} ${PP_FILES})

add_custom_target(preprocess DEPENDS ${PP_FILES})

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified. Setting it to Release.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(HWLOC_INCLUDE_DIR HWLOC_INCLUDE_DIR-NOTFOUND CACHE PATH "Location of hwloc.h")
set(LIBHWLOC LIBHWLOC-NOTFOUND) 

if (PKGCONFIG_FOUND AND NOT HWLOC_INCLUDE_DIR)
    message(STATUS "Try to find hwloc via pkg-config")
    find_package(HWLOC REQUIRED)
    find_library(LIBHWLOC hwloc)
endif()

# find hwloc.h in default header locations
if (NOT HWLOC_INCLUDE_DIR)
    find_path(HWLOC_INCLUDE_DIR hwloc.h)
endif()

if (NOT HWLOC_INCLUDE_DIR)
    message(FATAL_ERROR "HWLOC_INCLUDE_DIR not set!")
endif()

if (NOT LIBHWLOC)
    get_filename_component(HWLOC_LIBDIR ${HWLOC_INCLUDE_DIR}/../lib REALPATH)
    find_library(LIBHWLOC hwloc ${HWLOC_LIBDIR} NO_DEFAULT_PATH)
endif()

set(HWLOC_LIBRARIES ${LIBHWLOC})


mark_as_advanced(LIBHWLOC)
mark_as_advanced(CLEAR HWLOC_INCLUDE_DIR)

foreach(instr ${INSTRUMENTATION})
    if (${instr} STREQUAL "TIMING")
        set(GHOST_HAVE_INSTR_TIMING 1)
    endif()
    if (${instr} STREQUAL "LIKWID")
        set(GHOST_HAVE_INSTR_LIKWID 1)
        set(LIKWID_INCLUDE_DIR "" CACHE PATH "Location of likwid.h")
        include_directories(${LIKWID_INCLUDE_DIR})
        add_definitions(-DLIKWID_PERFMON)
        get_filename_component(LIKWID_LIBDIR ${LIKWID_INCLUDE_DIR}/../lib REALPATH)
        find_library(LIKWID_LIBRARIES likwid ${LIKWID_LIBDIR})
    endif()
endforeach()

if (USE_MPI)
    set(MPI_LIBRARIES "-lmpi_mt")
    find_package(MPI QUIET)
    include_directories(${MPI_INCLUDE_PATH})
    foreach(item ${MPI_LIBRARIES})
        if (BUILD_MIC)
            string(REPLACE "intel64" "mic" item_mic ${item})
            list(REMOVE_ITEM MPI_LIBRARIES ${item})
            list(APPEND MPI_LIBRARIES ${item_mic})
            set(item ${item_mic})
        endif()
        string(REGEX MATCH ".*libmpi.so" match ${item})
        if (match)
            get_filename_component(MPI_LIBDIR ${match} PATH)
            if (EXISTS "${MPI_LIBDIR}/libmpi_mt.so")
                list(REMOVE_ITEM MPI_LIBRARIES ${item})
                list(APPEND MPI_LIBRARIES ${MPI_LIBDIR}/libmpi_mt.so)
            endif()
        endif()
    endforeach()
    get_filename_component(MPI_LIBDIR "${MPI_LIBRARY}" PATH)
    find_library(LIBMPI_CXX mpi_cxx ${MPI_LIBDIR})
    set(GHOST_HAVE_MPI 1)
    mark_as_advanced(MPI_EXTRA_LIBRARY MPI_LIBRARY LIBMPI_CXX)
    mark_as_advanced(CLEAR MPI_C_INCLUDE_PATH)
endif()



if (USE_CUDA)
    find_package(CUDA)
    if (CUDA_FOUND)
        set(GHOST_HAVE_CUDA 1)
        cuda_include_directories(${PROJECT_SOURCE_DIR}/include
            ${PROJECT_BINARY_DIR} ${PROJECT_BINARY_DIR}/ghost ${HWLOC_INCLUDE_DIR} ${LIKWID_INCLUE_DIR} ${MPI_INCLUDE_PATH})
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC;-arch=sm_35)
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            list(APPEND CUDA_NVCC_FLAGS -O0)
        endif()
        if (USE_OPENMP)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fopenmp)
        endif()
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND CUDA_NVCC_FLAGS -G -g)
        endif()
        cuda_compile(SELL_KERNEL_CU_O ${PROJECT_BINARY_DIR}/src/sell_spmv.cu OPTIONS -DMPICH_IGNORE_CXX_SEEK)
        cuda_compile(DENSEMAT_CM_CU_O src/densemat_cm.cu OPTIONS -DMPICH_IGNORE_CXX_SEEK)
        cuda_compile(DENSEMAT_RM_CU_O src/densemat_rm.cu OPTIONS -DMPICH_IGNORE_CXX_SEEK)
        cuda_compile(CRS_CU_O src/crs.cu OPTIONS -DMPICH_IGNORE_CXX_SEEK)
        set (SOURCES 
            ${SOURCES}
            ${SELL_KERNEL_CU_O}
            ${DENSEMAT_CM_CU_O}
            ${DENSEMAT_RM_CU_O}
            ${CRS_CU_O}
            )
        include_directories(${CUDA_INCLUDE_DIRS})
        set (HEADERS ${HEADERS}
            include/ghost/cu_densemat_cm.h
            include/ghost/cu_densemat_rm.h
            )
        mark_as_advanced(
            CUDA_BUILD_CUBIN
            CUDA_BUILD_EMULATION
            CUDA_HOST_COMPILER
            CUDA_SDK_ROOT_DIR
            CUDA_SEPARABLE_COMPILATION
            CUDA_VERBOSE_BUILD
            )
    else ()
        message(WARNING "Could not find CUDA! You might want to make CUDA available or set USE_CUDA=OFF to suppress this warning.")
    endif()
endif()

if (${LIBTYPE} STREQUAL "STATIC")
    add_library(ghost STATIC ${SOURCES})
elseif (${LIBTYPE} STREQUAL "SHARED")
    add_library(ghost SHARED ${SOURCES})
else ()
    message(SEND_ERROR "Unknown library type. Setting it to SHARED")
    set(LIBTYPE "SHARED")
    add_library(ghost SHARED ${SOURCES})
endif()
add_dependencies(ghost preprocess)


if (USE_CUDA)
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${LIBCUDADEVRT})
endif()

if (USE_SCOTCH)
    set(GHOST_HAVE_SCOTCH 1)
    set(SCOTCH_INCLUDE_DIR "${SCOTCH_INCLUDE_DIR}" CACHE PATH "Location of PT-Scotch headers")
    if (USE_MPI)
        if (NOT EXISTS "${SCOTCH_INCLUDE_DIR}/ptscotch.h")
            message(FATAL_ERROR "Could not find ptscotch.h in SCOTCH_INCLUDE_DIR!")
        endif()
    else()
        if (NOT EXISTS "${SCOTCH_INCLUDE_DIR}/scotch.h")
            message(FATAL_ERROR "Could not find scotch.h in SCOTCH_INCLUDE_DIR!")
        endif()
    endif()
    include_directories(${SCOTCH_INCLUDE_DIR})
    get_filename_component(SCOTCH_LIBDIR ${SCOTCH_INCLUDE_DIR}/../lib REALPATH)

    set(LIBPTSCOTCH LIBPTSCOTCH-NOTFOUND)
    set(LIBSCOTCH LIBSCOTCH-NOTFOUND)
    set(LIBPTSCOTCHERR LIBPTSCOTCHERR-NOTFOUND)
    set(LIBSCOTCHERR LIBSCOTCHERR-NOTFOUND)

    find_library(LIBPTSCOTCH ptscotch ${SCOTCH_LIBDIR})
    find_library(LIBSCOTCH scotch ${SCOTCH_LIBDIR})
    find_library(LIBPTSCOTCHERR ptscotcherr ${SCOTCH_LIBDIR})
    find_library(LIBSCOTCHERR scotcherr ${SCOTCH_LIBDIR})
    if (USE_MPI)
        set(SCOTCH_LIBRARIES ${LIBPTSCOTCH} ${LIBSCOTCH} ${LIBPTSCOTCHERR})
    else()
        set(SCOTCH_LIBRARIES ${LIBSCOTCH} ${LIBSCOTCHERR})
    endif()
    mark_as_advanced(LIBPTSCOTCH LIBSCOTCH LIBPTSCOTCHERR LIBSCOTCHERR)
endif()


if (USE_COLPACK)
    set(GHOST_HAVE_COLPACK 1)
    set(COLPACK_INCLUDE_DIR "${COLPACK_INCLUDE_DIR}" CACHE PATH "Location of ColPack headers")
    include_directories(${COLPACK_INCLUDE_DIR})
    get_filename_component(COLPACK_LIBDIR ${COLPACK_INCLUDE_DIR}/../lib REALPATH)

    find_library(COLPACK_LIBRARIES ColPack ${COLPACK_LIBDIR})
    if (NOT COLPACK_LIBRARIES)
        message(FATAL_ERROR "COLPACK_INCLUDE_DIR not set correctly!")
    endif()
endif()

if (USE_OPENMP)	
    find_package(OpenMP REQUIRED)
    set(GHOST_HAVE_OPENMP 1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CBLAS_INCLUDE_DIR CBLAS_INCLUDE_DIR-NOTFOUND CACHE PATH "Location of *cblas.h")

# firstly, try to find a Cblas header in the default locations
if (NOT CBLAS_INCLUDE_DIR)
    message(STATUS "Try to find MKL in default paths and via MKLROOT")
    find_path(CBLAS_INCLUDE_DIR mkl_cblas.h $ENV{MKLROOT}/include)
endif()
if (NOT CBLAS_INCLUDE_DIR)
    #Have to search GSL with find_package because it will not be found by find_path
    message(STATUS "Try to find GSL in default paths and via find_package")
    find_package(GSL)
    find_path(CBLAS_INCLUDE_DIR gsl_cblas.h ${GSL_INCLUDE_DIRS}/gsl)
endif()
if (NOT CBLAS_INCLUDE_DIR)
    message(STATUS "Try to find default Cblas")
    find_path(CBLAS_INCLUDE_DIR cblas.h)
endif()

if (NOT CBLAS_INCLUDE_DIR)
    message(FATAL_ERROR "CBLAS_INCLUDE_DIR not set!")
endif()

include_directories(SYSTEM ${CBLAS_INCLUDE_DIR})

file(GLOB CBLAS_H_FULL "${CBLAS_INCLUDE_DIR}/*cblas.h")
get_filename_component(CBLAS_H ${CBLAS_H_FULL} NAME)

if (${CBLAS_H} STREQUAL "mkl_cblas.h")
    message(STATUS "MKL BLAS found")
    set(GHOST_HAVE_MKL 1)
    set(GHOST_HAVE_LAPACK 1)
    if (BUILD_MIC)
        get_filename_component(MKL_LIBDIR ${CBLAS_INCLUDE_DIR}/../lib/mic REALPATH)
    else ()
        get_filename_component(MKL_LIBDIR ${CBLAS_INCLUDE_DIR}/../lib/intel64 REALPATH)
    endif()

    set(LIBMKL_CORE LIBMKL_CORE-NOTFOUND)
    set(LIBMKL_X_THREAD LIBMKL_X_THREAD-NOTFOUND)
    set(LIBMKL_INTEL_XLP64 LIBMKL_INTEL_XLP64-NOTFOUND)

    find_library(LIBMKL_CORE mkl_core ${MKL_LIBDIR})

    if (USE_LONGIDX_LOCAL)
        add_definitions(-DMKL_ILP64)
        find_library(LIBMKL_INTEL_XLP64 mkl_intel_ilp64 ${MKL_LIBDIR})
    else()
        find_library(LIBMKL_INTEL_XLP64 mkl_intel_lp64 ${MKL_LIBDIR})
    endif()

    if(USE_OPENMP)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel") 
            find_library(LIBMKL_X_THREAD mkl_intel_thread ${MKL_LIBDIR})
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") 
            find_library(LIBMKL_X_THREAD mkl_gnu_thread ${MKL_LIBDIR})
        else ()
            message(FATAL_ERROR "OpenMP only possible with Intel or Gnu!")
        endif()
    else ()
        find_library(LIBMKL_X_THREAD mkl_sequential ${MKL_LIBDIR})
    endif()

    set (BLAS_LIBRARIES "${LIBMKL_CORE};${LIBMKL_INTEL_XLP64};${LIBMKL_X_THREAD}" CACHE STRING "BLAS libraries")
    mark_as_advanced(LIBMKL_CORE LIBMKL_INTEL_XLP64 LIBMKL_X_THREAD) 
elseif (${CBLAS_H} STREQUAL "gsl_cblas.h")
    set(GHOST_HAVE_GSL 1)
    get_filename_component(GSL-CONFIG_PATH ${CBLAS_INCLUDE_DIR}/../../bin REALPATH)
    list(APPEND CMAKE_PROGRAM_PATH ${GSL-CONFIG_PATH})
    find_package(GSL REQUIRED)
    message(STATUS "GSL BLAS found")
    set (BLAS_LIBRARIES "${GSL_LIBRARIES}" CACHE STRING "BLAS libraries")
else () #generic BLAS
    set (BLAS_LIBRARIES "" CACHE STRING "BLAS libraries")
    message(STATUS "Generic BLAS. BLAS_LIBRARIES has to be set manually before linking.")
endif()

set(LAPACK_INCLUDE_DIR CACHE PATH "Location of lapacke.h")

if (NOT LAPACK_INCLUDE_DIR)
    message(STATUS "Determine LAPACK...")

    if (${CBLAS_H} STREQUAL "mkl_cblas.h")
        message(STATUS "MKL LAPACK is used")
        set(GHOST_HAVE_LAPACK 1)
        set(LAPACK_LIBRARIES ${BLAS_LIBRARIES})
        find_path(LAPACK_INCLUDE_DIR mkl_lapacke.h ${CBLAS_INCLUDE_DIR})
    else()
        message(STATUS "Try to find LAPACK")
        find_package(LAPACK)
        if (LAPACK_FOUND)
            message(STATUS "LAPACK found!")
            set(GHOST_HAVE_LAPACK 1)
            include_directories(LAPACK_INCLUDE_DIR)

            string(REPLACE "lapack.so" "lapacke.so" LAPACK_LIBRARIES "${LAPACK_LIBRARIES}")
        else()
            message(WARNING "LAPACK not found! Some functionality will not work!")
            set(LAPACK_LIBRARIES "")
        endif()
    endif()
endif()

if (NOT LAPACK_LIBRARIES AND LAPACK_INCLUDE_DIR)
    message(STATUS "LAPACK include dir has been set. Automatically setting LAPACK libraries...")
    get_filename_component(LAPACK_LIBRARIES ${LAPACK_INCLUDE_DIR}/../lib/liblapacke.so REALPATH)
    message(STATUS "LAPACK libraries set to: ${LAPACK_LIBRARIES}")
    set(GHOST_HAVE_LAPACK 1)
endif()

include_directories(${LAPACK_INCLUDE_DIR})    
mark_as_advanced(CLEAR LAPACK_INCLUDE_DIR)
mark_as_advanced(CLEAR LAPACK_LIBRARIES)

if (NOT EXISTS "${HWLOC_INCLUDE_DIR}/hwloc.h")
    message(FATAL_ERROR "Could not find hwloc.h neither via pkg-config nor in the defined HWLOC_INCLUDEDIR!")
endif()

include_directories(BEFORE ${HWLOC_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/ghost ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
    # msg 193 complains about commented out lines in config.h
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -h c99 -h msglevel_3 -h nomessage=193")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -h msglevel_3 -h nomessage=193")
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -std=c99 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++0x -Wall -Wextra -pedantic")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-variable -save-temps -fsource-asm")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable -save-temps -fsource-asm")
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-alias")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-alias")
        endif()
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND NOT "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "4.8")) 
            if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -fstrict-aliasing")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fstrict-aliasing")
            elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
                set(CLANG_SANITIZE "" CACHE STRING "What should be sanitized by Clang/GCC for a Debug build? (Adress, Thread, Memory, Undefined or leave empty)")
                if ("${CLANG_SANITIZE}" STREQUAL "Address")
                    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
                elseif ("${CLANG_SANITIZE}" STREQUAL "Thread")
                    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread -pie")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -pie")
                elseif ("${CLANG_SANITIZE}" STREQUAL "Memory")
                    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
                      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
                      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
                    else()
                      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak -fno-omit-frame-pointer")
                      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fno-omit-frame-pointer")
                    endif()
                elseif ("${CLANG_SANITIZE}" STREQUAL "Undefined")
                    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=undefined")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
                endif()
            endif()
        endif()
    endif()
    # -fstrict-aliasing is set in GCC with -O3
endif ()
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif()



if (BUILD_MIC)
    set(GHOST_HAVE_MIC 1)
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        message(FATAL_ERROR "The ${CMAKE_CXX_COMPILER_ID} compiler cannot build MIC code. Specify a suited compiler or set BUILD_MIC=OFF!")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
else()
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
        AutodetectHostArchitecture()
        OptimizeForArchitecture()
        if (AVX_FOUND)
            set(GHOST_HAVE_AVX 1)
        endif()
        if (SSE2_FOUND OR SSE3_FOUND OR SSE4_1_FOUND OR SSE4_2_FOUND OR SSE4a_FOUND)
            set(GHOST_HAVE_SSE 1)
        endif()

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
    endif()
endif()

configure_file (
    "${PROJECT_SOURCE_DIR}/include/ghost/config.h.in"
    "${PROJECT_BINARY_DIR}/ghost/config.h"
    )
configure_file (
    "${PROJECT_SOURCE_DIR}/include/ghost/func_util.h.in"
    "${PROJECT_BINARY_DIR}/ghost/func_util.h"
    )
configure_file(${PROJECT_SOURCE_DIR}/ghost.pc.in ${PROJECT_BINARY_DIR}/ghost.pc @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/ghost-config.cmake.in ${PROJECT_BINARY_DIR}/ghost-config.cmake @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/ghost-config-version.cmake.in ${PROJECT_BINARY_DIR}/ghost-config-version.cmake @ONLY)

if (${LIBTYPE} STREQUAL "STATIC")
    install(TARGETS ghost EXPORT ghost-targets ARCHIVE DESTINATION lib/ghost)
elseif (${LIBTYPE} STREQUAL "SHARED")
    install(TARGETS ghost EXPORT ghost-targets LIBRARY DESTINATION lib/ghost)
endif()
install(FILES ${PROJECT_BINARY_DIR}/ghost-config.cmake DESTINATION lib/ghost)
install(FILES ${PROJECT_BINARY_DIR}/ghost-config-version.cmake DESTINATION lib/ghost)
install(FILES include/ghost.h DESTINATION include)
install(FILES ${HEADERS} DESTINATION include/ghost)
install(FILES ${PROJECT_BINARY_DIR}/ghost.pc DESTINATION lib/pkgconfig)
install(FILES ${PROJECT_BINARY_DIR}/ghost/config.h DESTINATION include/ghost)
install(FILES ${PROJECT_BINARY_DIR}/ghost/func_util.h DESTINATION include/ghost)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)
add_custom_target(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake") 

find_package(Doxygen)
if (DOXYGEN_FOUND)
    configure_file(${PROJECT_SOURCE_DIR}/doxygen/mainpage.md.in ${PROJECT_BINARY_DIR}/doxygen/mainpage.md @ONLY)
    configure_file(${PROJECT_SOURCE_DIR}/doxygen/Doxyfile.in ${PROJECT_BINARY_DIR}/doxygen/Doxyfile @ONLY)
    set(DOC_INSTALL_PREFIX "" CACHE PATH "Where to store the generated documentation")
    mark_as_advanced(DOC_INSTALL_PREFIX)
    file(GLOB DOCFILES "${PROJECT_SOURCE_DIR}/doxygen/[^.]*")
    if (EXISTS ${DOC_INSTALL_PREFIX})
        add_custom_command(
            OUTPUT ${PROJECT_BINARY_DIR}/doc/html/index.html 
            DEPENDS ${HEADERS} ${DOCFILES}
            COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doxygen/Doxyfile && 
            cp ${PROJECT_SOURCE_DIR}/contrib/ghost_logo.svg ${PROJECT_BINARY_DIR}/doc/html && 
            rsync -a ${PROJECT_BINARY_DIR}/doc/html/. ${DOC_INSTALL_PREFIX} && 
            find ${DOC_INSTALL_PREFIX} -type d -exec chmod 755 {} + &&
            find ${DOC_INSTALL_PREFIX} -type f -exec chmod 644 {} +
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
            )
    else ()
        add_custom_command(
            OUTPUT ${PROJECT_BINARY_DIR}/doc/html/index.html 
            DEPENDS ${SOURCES} ${HEADERS} ${DOCFILES} 
            COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doxygen/Doxyfile && cp ${PROJECT_SOURCE_DIR}/contrib/ghost_logo.svg ${PROJECT_BINARY_DIR}/doc/html 
            )
    endif()

    add_custom_target(doc DEPENDS ${PROJECT_BINARY_DIR}/doc/html/index.html)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/test)
enable_testing()
add_test(NAME simple WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test COMMAND simple)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --output-on-failure DEPENDS simple)

