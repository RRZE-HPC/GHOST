cmake_minimum_required (VERSION 2.8)
project (GHOST)

set(GHOST_VERSION_MAJOR 1)
set(GHOST_VERSION_MINOR 0)
set(GHOST_VERSION_REVISION 5)
set(GHOST_VERSION_IDENTIFIER)
set(GHOST_VERSION "${GHOST_VERSION_MAJOR}.${GHOST_VERSION_MINOR}.${GHOST_VERSION_REVISION}")
if (DEFINED GHOST_VERSION_IDENTIFIER)
    set(GHOST_VERSION "${GHOST_VERSION}-${GHOST_VERSION_IDENTIFIER}")
endif()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

include(CheckCCompilerFlag)
include(${PROJECT_SOURCE_DIR}/cmake/OptimizeForArchitecture.cmake)

option(GHOST_USE_MPI "Build with MPI" 1)
option(GHOST_USE_OPENMP "Build with OpenMP" 1)
option(GHOST_USE_CUDA "Build with CUDA" 1)
option(GHOST_USE_SCOTCH "Use PT-Scotch for matrix re-ordering" 0)
option(GHOST_USE_COLPACK "Use ColPack for matrix coloring" 0)
option(GHOST_USE_ZOLTAN "Use Zoltan for matrix re-ordering" 0)
option(GHOST_USE_SPMP "Use SpMP for matrix preprocessing" 0)
option(GHOST_IDX64_LOCAL "Enable 64-bit indices for local quantities" 0)
option(GHOST_IDX64_GLOBAL "Enable 64-bit indices for global quantities" 1)
option(GHOST_OPTIMIZE_HOSTARCH "Create optimized code for the host's architecture" 1)
option(GHOST_BUILD_MIC "Build for Intel MIC" 0)
option(GHOST_LOG_ONLYFIRST "Print all errors, warnings and info only once for each occurence" 1)
option(GHOST_TRACK_DATATRANSFERS "Track MPI and CUDA data transfers" 0)
option(GHOST_SPARSEMAT_GLOBALSTATS "Gather some global statistics on sparse matrices (should be disabled for large-scale runs)" 0)
option(GHOST_SPARSEMAT_STATS "Gather statistics about sparse matrices (slows down matrix construction)" 0)
option(GHOST_KACZ_ANALYZE "Gather data about load balancing for KACZ method (might slow down initial splitting)" 0)  
set(GHOST_LOG_RANK "-1" CACHE STRING "The MPI rank for which the log messages will be printed. -1 for full verbosity.")
set(GHOST_GEN_DENSEMAT_DIM "1,2,4,8" CACHE STRING "The block vector sizes for which generated kernels are being created (comma-separated list)")
set(GHOST_GEN_SELL_C "1,32" CACHE STRING "The SELL chunk heights for which generated kernels are being created (comma-separated list)")
set(GHOST_GEN_TSM_UNROLL "2" CACHE STRING "Outer loop unrolling dimensions to be generated for tall&skinny GEMM kernels")
set(GHOST_GEN_CUSELLSPMV "true,true,true,true,true,true,true;true,false,false,false,false,false,false;false,false,false,false,false,false,false" CACHE STRING "The permutations of SELL SpMV CUDA kernels to instantiate (semicolon-separated list of comma-separated lists). AXP(B)Y,SCALE,(V)SHIFT,DOT_YY,DOT_XY,DOT_XX,CHAIN_AXPBY")
set(GHOST_VERBOSITY 1 CACHE STRING "Level of verbosity information. Valid values: 0 (be totally silent), 1 (print warnings, errors and further important information), 2, 3, 4 (increasing level of verbosity)")
set(GHOST_INSTRUMENT "" CACHE STRING "Type of instrumentation (semicolon separated list). Valid values: TIMING, LIKWID, IACA")
set(GHOST_INSTRUMENT_FUNCTYPES "(functype) & GHOST_FUNCTYPE_ANY" CACHE STRING "Logical selection of function types to instrument. Possible values are defined in func_util.h and functype should be put in parentheses.")
option(GHOST_INSTRUMENT_BARRIER "Place a barrier after each instrumented function (may be necessary for correct timings)" 0)

find_package(PkgConfig)

add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX)
if (GHOST_IDX64_LOCAL)
    if (NOT GHOST_IDX64_GLOBAL)
        message(FATAL_ERROR "Having 64-bit indices locally and only 32-bit indices globally is not allowed (and not sensible)!")
    endif()
endif()

set (SOURCES
    src/datatransfers.cpp 
    src/densemat_cm_averagehalo.cpp
    src/densemat_cm.cpp 
    src/densemat_rm_averagehalo.cpp
    src/densemat_rm.cpp
    src/dot.cpp 
    src/errorhandler.cpp
    src/sell_kacz.cpp
    src/sell_spmv.cpp
    src/sparsemat.cpp
    src/sparsemat_perm_colpack.cpp
    src/sparsemat_perm_spmp.cpp
    src/timing.cpp
    src/tsmm.cpp
    src/tsmm_inplace.cpp
    src/tsmttsm.cpp
    src/bench.c
    src/bincrs_func.c
    src/bitmap.c
    src/blas_util.c
    src/carp.c
    src/context.c
    src/core.c
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
    src/perm.c
    src/pumap.c
    src/rand.c
    src/sparsemat.c
    src/sparsemat_perm_scotch.c
    src/sparsemat_perm_sort.c
    src/sparsemat_perm_zoltan.c
    src/spmv.c
    src/spmv_overlap.c 
    src/spmv_nompi.c 
    src/spmv_task.c
    src/spmv_vector.c 
    src/task.c
    src/taskq.c
    src/thpool.c
    src/timing.c
    src/types.c
    src/util.c
    src/rcm_dissection.cpp
    src/sell_kacz_rb.c
    src/carp_rb.c
    src/sell_kacz_rb_versions.c 
    src/sparsemat_write.cpp
    src/sparsemat_perm_blockColor.cpp 
    src/kacz_hybrid_split.c
    src/kacz_split_analytical.c
    src/sell_kacz_bmc.c 
    src/sell_kacz_mc.c
    src/sell_kacz_selector.c
    src/kacz_analyze_print.c
    )

set (HEADERS
    include/ghost/bench.h
    include/ghost/bincrs.h
    include/ghost/bindensemat.h
    include/ghost/bitmap.h
    include/ghost/blas_mangle.h
    include/ghost/carp.h
    include/ghost/complex.h
    include/ghost/context.h
    include/ghost/core.h
    include/ghost/cu_bench.h
    include/ghost/cu_util.h
    include/ghost/datatransfers.h
    include/ghost/densemat.h
    include/ghost/densemat_cm.h
    include/ghost/densemat_iter_macros.h
    include/ghost/densemat_rm.h
    include/ghost/dot.h
    include/ghost/error.h
    include/ghost/errorhandler.h
    include/ghost/funcptr_wrappers.h
    include/ghost/helper.h
    include/ghost/instr.h
    include/ghost/locality.h	
    include/ghost/log.h
    include/ghost/machine.h
    include/ghost/math.h
    include/ghost/matrixmarket.h
    include/ghost/omp.h
    include/ghost/pumap.h
    include/ghost/rand.h
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
    include/ghost/rcm_dissection.h
    include/ghost/sell_kacz_rb.h
    include/ghost/carp_rb.h
    include/ghost/sell_kacz_rb_versions.h 
    include/ghost/kacz_hybrid_split.h
    include/ghost/kacz_split_analytical.h
    include/ghost/sell_kacz_bmc.h 
    include/ghost/sell_kacz_mc.h
    )

string(REPLACE "," ";" GHOST_GEN_SELL_C_LIST ${GHOST_GEN_SELL_C})
string(REPLACE "," ";" GHOST_GEN_DENSEMAT_DIM_LIST ${GHOST_GEN_DENSEMAT_DIM})
string(REPLACE "," ";" GHOST_GEN_UNROLL_LIST ${GHOST_GEN_TSM_UNROLL})

set(SELL_CU_SPMV_SWITCHCASCADE "bool do_fallback = false;\\\nswitch \(mat->traits.C) {\\\n")
foreach (SELL_CHUNKHEIGHT ${GHOST_GEN_SELL_C_LIST})
    set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}    case ${SELL_CHUNKHEIGHT}:\\\n")
    set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}        switch \(rhs->traits.ncols) {\\\n")
    foreach (SELL_NCOLS ${GHOST_GEN_DENSEMAT_DIM_LIST})
        set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}            case ${SELL_NCOLS}:\\\n")
        foreach(SELL_CUDA_VARIANT ${GHOST_GEN_CUSELLSPMV})
            set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}                if(!(strcmp(boolstr,\"${SELL_CUDA_VARIANT}\"))) { kernel<dt1,dt2,dt2_b,nrowsinblock,${SELL_CHUNKHEIGHT},${SELL_NCOLS},${SELL_CUDA_VARIANT}><<<grid,block,smem>>>((dt2 *)lhsval,(int)(lhs->stride),(dt2 *)rhsval,(int)rhs->stride,traits.flags,mat->nrows,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,(dt2 *)cu_shift,(dt2)scale,(dt2)beta,(dt2 *)cu_localdot,(dt2 *)zval,zstride,(dt2)sdelta,(dt2)seta); } else \\\n")
        endforeach()
        set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}                { PERFWARNING_LOG(\"Non-configured bool variant: %s\",boolstr);\\\n                do_fallback = true; }\\\n")
        set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}        break;\\\n")
    endforeach() 
    set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}            default:\\\n                PERFWARNING_LOG(\"Non-configured ncols %d!\",rhs->traits.ncols);\\\n                do_fallback = true;\\\n        }\\\n        break;\\\n")
endforeach()
set(SELL_CU_SPMV_SWITCHCASCADE "${SELL_CU_SPMV_SWITCHCASCADE}    default:\\\n        PERFWARNING_LOG(\"Non-configured chunk height: %d!\",mat->traits.C);\\\n        do_fallback = true;\\\n}\\\n")

set (PP_SCRIPT ${PROJECT_SOURCE_DIR}/bin/ghost_unroll.pl)
set (PP_HEADER_SCRIPT ${PROJECT_SOURCE_DIR}/bin/ghost_extractfunc.pl)
set (PP_KERNELMAP_SCRIPT ${PROJECT_SOURCE_DIR}/bin/ghost_mapfunc.pl)
set (UNROLLED_SRC_DIR ${PROJECT_BINARY_DIR}/src_unrolled/)
set (GENERATED_INC_DIR ${PROJECT_BINARY_DIR}/ghost/)

file(GLOB TEMPLATE_SRCFILES "${CMAKE_SOURCE_DIR}/src/*.in")
foreach(TEMPLATE_SRCFILE ${TEMPLATE_SRCFILES})
    set(GENERATED_FILES "")
    set(UNROLLED_FILES "")
    get_filename_component(AUTOGEN_FUNCTION ${TEMPLATE_SRCFILE} NAME_WE)
    get_filename_component(TEMPLATE_SRCFILE_FULLEXT ${TEMPLATE_SRCFILE} EXT)
    string(REPLACE ".in" "" TEMPLATE_SRCFILE_EXT ${TEMPLATE_SRCFILE_FULLEXT})

    file(READ ${TEMPLATE_SRCFILE} TEMPLATE_HEADER OFFSET 0 LIMIT 17)
    string(STRIP ${TEMPLATE_HEADER} TEMPLATE_HEADER_STRIPPED)
    if (NOT "${TEMPLATE_HEADER_STRIPPED}" STREQUAL "/*!GHOST_AUTOGEN")
        message(FATAL_ERROR "AUTOGEN header missing or wrong in file ${TEMPLATE_SRCFILE}. Is ${TEMPLATE_HEADER}, should be /*!GHOST_AUTOGEN")
    endif()

    file(READ ${TEMPLATE_SRCFILE} TEMPLATE_DEPS_RAW OFFSET 17 LIMIT 31)
    string(FIND "${TEMPLATE_DEPS_RAW}" " " ENDPOS)
    string(SUBSTRING "${TEMPLATE_DEPS_RAW}" 0 ${ENDPOS} TEMPLATE_DEPS)
    set(TEMPLATE_DEPS_LIST ${TEMPLATE_DEPS})
   
    if (DEFINED TEMPLATE_DEPS_LIST)
        list(SORT TEMPLATE_DEPS_LIST)
    endif()
   
    if(TEMPLATE_DEPS_LIST MATCHES "^BLOCKDIM1$")
        foreach(BLOCKDIM1 ${GHOST_GEN_DENSEMAT_DIM_LIST})
            configure_file (
                ${TEMPLATE_SRCFILE}
                ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}__${BLOCKDIM1}${TEMPLATE_SRCFILE_EXT}
                )
            list(APPEND GENERATED_FILES ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}__${BLOCKDIM1}${TEMPLATE_SRCFILE_EXT})
        endforeach()
    elseif(TEMPLATE_DEPS_LIST MATCHES "^CHUNKHEIGHT$")
        foreach(CHUNKHEIGHT ${GHOST_GEN_SELL_C_LIST})
configure_file (
                ${TEMPLATE_SRCFILE}
                ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}__${CHUNKHEIGHT}${TEMPLATE_SRCFILE_EXT}
                )
            list(APPEND GENERATED_FILES ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}__${CHUNKHEIGHT}${TEMPLATE_SRCFILE_EXT})
        endforeach()
    elseif(TEMPLATE_DEPS_LIST MATCHES "^BLOCKDIM1;CHUNKHEIGHT$")
        foreach(BLOCKDIM1 ${GHOST_GEN_DENSEMAT_DIM_LIST})
            foreach(CHUNKHEIGHT ${GHOST_GEN_SELL_C_LIST})
                configure_file (
                    ${TEMPLATE_SRCFILE}
                    ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}__${BLOCKDIM1}_${CHUNKHEIGHT}${TEMPLATE_SRCFILE_EXT}
                    )
                list(APPEND GENERATED_FILES ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}__${BLOCKDIM1}_${CHUNKHEIGHT}${TEMPLATE_SRCFILE_EXT})
            endforeach()
        endforeach()
    elseif(TEMPLATE_DEPS_LIST MATCHES "^BLOCKDIM1;BLOCKDIM2$")
        foreach(BLOCKDIM1 ${GHOST_GEN_DENSEMAT_DIM_LIST})
            foreach(BLOCKDIM2 ${GHOST_GEN_DENSEMAT_DIM_LIST})
                configure_file (
                    ${TEMPLATE_SRCFILE}
                    ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}__${BLOCKDIM1}_${BLOCKDIM2}${TEMPLATE_SRCFILE_EXT}
                    )
                list(APPEND GENERATED_FILES ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}__${BLOCKDIM1}_${BLOCKDIM2}${TEMPLATE_SRCFILE_EXT})
            endforeach()
        endforeach()
    elseif(TEMPLATE_DEPS_LIST MATCHES "^BLOCKDIM1;BLOCKDIM2;UNROLL$")
        foreach(BLOCKDIM1 ${GHOST_GEN_DENSEMAT_DIM_LIST})
            foreach(BLOCKDIM2 ${GHOST_GEN_DENSEMAT_DIM_LIST})
                foreach(UNROLL ${GHOST_GEN_UNROLL_LIST})
                    configure_file (
                        ${TEMPLATE_SRCFILE}
                        ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}__${BLOCKDIM1}_${BLOCKDIM2}_${UNROLL}${TEMPLATE_SRCFILE_EXT}
                        )
                    list(APPEND GENERATED_FILES ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}__${BLOCKDIM1}_${BLOCKDIM2}_${UNROLL}${TEMPLATE_SRCFILE_EXT})
                endforeach()
            endforeach()
        endforeach()
    elseif(TEMPLATE_DEPS_LIST MATCHES "")
        configure_file (
            ${TEMPLATE_SRCFILE}
            ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}${TEMPLATE_SRCFILE_EXT}
            )
        list(APPEND GENERATED_FILES ${PROJECT_BINARY_DIR}/src_generated/${AUTOGEN_FUNCTION}${TEMPLATE_SRCFILE_EXT})
    else ()
        message(FATAL_ERROR "Unsupported AUTOGEN dependency list: ${TEMPLATE_DEPS_LIST}")
    endif()

    foreach (TO_UNROLL_FILE ${GENERATED_FILES})
        get_filename_component(TO_UNROLL_FILENAME ${TO_UNROLL_FILE} NAME)
        get_filename_component(TO_UNROLL_FILENAME_WE ${TO_UNROLL_FILE} NAME_WE)
        set(UNROLLED_FILE ${UNROLLED_SRC_DIR}/${TO_UNROLL_FILENAME})
        add_custom_command(
            OUTPUT ${UNROLLED_FILE}
            DEPENDS ${TO_UNROLL_FILE} ${UNROLL_SCRIPT} 
            COMMAND mkdir -p ${UNROLLED_SRC_DIR} && perl ${PP_SCRIPT} ${TO_UNROLL_FILE} > ${UNROLLED_FILE}
            )
        list(APPEND UNROLLED_FILES ${UNROLLED_FILE})
    endforeach()

    set(GENERATED_HEADER ${GENERATED_INC_DIR}/${AUTOGEN_FUNCTION}_gen.h)
    set(GENERATED_KERNELMAP ${GENERATED_INC_DIR}/${AUTOGEN_FUNCTION}.def)
    add_custom_command(
        OUTPUT ${GENERATED_HEADER}
        DEPENDS ${UNROLLED_FILES} ${PP_HEADER_SCRIPT} 
        COMMAND mkdir -p ${GENERATED_INC_DIR} && perl ${PP_HEADER_SCRIPT} ${UNROLLED_FILES} > ${GENERATED_HEADER}
        )
    add_custom_command(
        OUTPUT ${GENERATED_KERNELMAP}
        DEPENDS ${GENERATED_HEADER} ${PP_KERNELMAP_SCRIPT} 
        COMMAND mkdir -p ${GENERATED_INC_DIR} && perl ${PP_KERNELMAP_SCRIPT} ${GENERATED_HEADER} > ${GENERATED_KERNELMAP}
        )
    list(APPEND HEADERS ${GENERATED_HEADER})
    if (${TEMPLATE_SRCFILE_EXT} STREQUAL ".cu")
        list(APPEND CU_GENERATED_SOURCES ${UNROLLED_FILES})
    else()
        list(APPEND SOURCES ${UNROLLED_FILES})
    endif()
    list(APPEND GLOB_UNROLLED_FILES ${UNROLLED_FILES})
    list(APPEND GLOB_GENERATED_FILES ${GENERATED_FILES})
    list(APPEND GLOB_GENERATED_HEADERS ${GENERATED_HEADER})
    list(APPEND GLOB_GENERATED_KERNELMAPS ${GENERATED_KERNELMAP})
    set(GLOB_UNROLLED_FILES ${GLOB_UNROLLED_FILES} ${UNROLLED_FILES})
endforeach()

add_custom_target(autogen DEPENDS ${GLOB_UNROLLED_FILES} ${GLOB_GENERATED_FILES} ${GLOB_GENERATED_HEADERS} ${GLOB_GENERATED_KERNELMAPS})

#set_source_files_properties(${PP_SRC_DIR}/tsmttsm_kahan.c PROPERTIES COMPILE_FLAGS "-fp-model strict")

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified. Setting it to Release.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if (NOT BUILD_SHARED_LIBS)
    message(STATUS "No library type specified. Setting it to shared.")
    set(BUILD_SHARED_LIBS ON CACHE BOOL "Set to ON/OFF for shared/static library build.")
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
    find_library(LIBHWLOC hwloc HINTS ${HWLOC_LIBDIR})
endif()

set(HWLOC_LIBRARIES ${LIBHWLOC})


mark_as_advanced(LIBHWLOC)
mark_as_advanced(CLEAR HWLOC_INCLUDE_DIR)

set(GHOST_INSTR_LIKWID 0)
set(GHOST_INSTR_TIMING 0)

foreach(instr ${GHOST_INSTRUMENT})
    if (${instr} STREQUAL "TIMING")
        set(GHOST_INSTR_TIMING 1)
    endif()
    if (${instr} STREQUAL "LIKWID")
        set(GHOST_INSTR_LIKWID 1)
        set(LIKWID_INCLUDE_DIR "" CACHE PATH "Location of likwid.h")
        include_directories(${LIKWID_INCLUDE_DIR})
        get_filename_component(LIKWID_LIBDIR ${LIKWID_INCLUDE_DIR}/../lib REALPATH)
        find_library(LIKWID_LIBRARIES likwid ${LIKWID_LIBDIR})
    endif()
    if (${instr} STREQUAL "IACA")
        message(STATUS "Disabling OpenMP because IACA instrumentation is enabled")
        set(GHOST_INSTR_IACA 1)
        set(GHOST_USE_OPENMP OFF)
    endif()
endforeach()

if (NOT GHOST_INSTR_IACA)
    add_definitions(-DIACA_MARKS_OFF)
endif()

if (GHOST_INSTRUMENT_BARRIER)
    set (GHOST_INSTR_BARRIER 1)
else()
    set (GHOST_INSTR_BARRIER 0)
endif()

if (GHOST_USE_MPI)
    set(MPI_LIBRARIES "-lmpi_mt")
    find_package(MPI QUIET)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    foreach(item ${MPI_LIBRARIES})
        if (GHOST_BUILD_MIC)
            string(REPLACE "intel64" "mic" item_mic ${item})
            list(REMOVE_ITEM MPI_LIBRARIES ${item})
            list(APPEND MPI_LIBRARIES ${item_mic})
            set(item ${item_mic})
        endif()
        string(REGEX MATCH ".*libmpi.so" match ${item})
        if (match)
            get_filename_component(MPI_LIBDIR ${match} PATH)
            if (EXISTS "${MPI_LIBDIR}/release_mt/")
                list(REMOVE_ITEM MPI_LIBRARIES ${item})
                list(APPEND MPI_LIBRARIES ${MPI_LIBDIR}/release_mt/libmpi.so)
            endif()
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
else ()
    set(GHOST_HAVE_MPI 0)
endif()



if (GHOST_USE_CUDA)
    option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    find_package(CUDA)
    if (NOT CUDA_FOUND)
        message(STATUS "Could not find CUDA!")
        set(GHOST_USE_CUDA 0)
        set(CUDA_LIBRARIES "")
    else ()
        set(GHOST_HAVE_CUDA 1)
        cuda_include_directories(${PROJECT_SOURCE_DIR}/include
            ${PROJECT_BINARY_DIR} ${PROJECT_BINARY_DIR}/ghost ${HWLOC_INCLUDE_DIR} ${LIKWID_INCLUE_DIR} ${MPI_INCLUDE_PATH})
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC;-arch=sm_35)
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            list(APPEND CUDA_NVCC_FLAGS -O0)
        endif()
        if (GHOST_USE_OPENMP)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fopenmp)
        endif()
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND CUDA_NVCC_FLAGS -G -g)
        endif()
        file(GLOB CU_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cu")
        cuda_compile(CU_OBJS ${CU_SOURCES})
        cuda_compile(CU_GENERATED_OBJS ${CU_GENERATED_SOURCES})
        
        set (SOURCES 
            ${SOURCES}
            ${CU_GENERATED_OBJS}
            ${CU_OBJS})
        include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
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
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
    endif()
endif()

add_library(ghost ${SOURCES})
add_dependencies(ghost autogen)

if (GHOST_USE_SCOTCH)
    set(GHOST_HAVE_SCOTCH 1)
    set(SCOTCH_INCLUDE_DIR "${SCOTCH_INCLUDE_DIR}" CACHE PATH "Location of PT-Scotch headers")
    if (GHOST_USE_MPI)
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
    if (GHOST_USE_MPI)
        set(SCOTCH_LIBRARIES ${LIBPTSCOTCH} ${LIBSCOTCH} ${LIBPTSCOTCHERR})
    else()
        set(SCOTCH_LIBRARIES ${LIBSCOTCH} ${LIBSCOTCHERR})
    endif()
    mark_as_advanced(LIBPTSCOTCH LIBSCOTCH LIBPTSCOTCHERR LIBSCOTCHERR)
endif()

if (GHOST_USE_ZOLTAN)
    set(GHOST_HAVE_ZOLTAN 1)
    set(ZOLTAN_INCLUDE_DIR "${ZOLTAN_INCLUDE_DIR}" CACHE PATH "Location of Zoltan headers")
    include_directories(${ZOLTAN_INCLUDE_DIR})
    get_filename_component(ZOLTAN_LIBDIR ${ZOLTAN_INCLUDE_DIR}/../lib REALPATH)

    find_library(ZOLTAN_LIBRARIES zoltan ${ZOLTAN_LIBDIR})
    if (NOT ZOLTAN_LIBRARIES)
        message(FATAL_ERROR "ZOLTAN_INCLUDE_DIR not set correctly!")
    endif()
endif()


if (GHOST_USE_COLPACK)
    set(GHOST_HAVE_COLPACK 1)
    set(COLPACK_INCLUDE_DIR "${COLPACK_INCLUDE_DIR}" CACHE PATH "Location of ColPack headers")
    include_directories(SYSTEM ${COLPACK_INCLUDE_DIR})
    get_filename_component(COLPACK_LIBDIR ${COLPACK_INCLUDE_DIR}/../lib REALPATH)

    find_library(COLPACK_LIBRARIES ColPack ${COLPACK_LIBDIR})
    if (NOT COLPACK_LIBRARIES)
        message(FATAL_ERROR "COLPACK_INCLUDE_DIR not set correctly!")
    endif()
endif()

if (GHOST_USE_OPENMP)	
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
    if (GHOST_BUILD_MIC)
        get_filename_component(MKL_LIBDIR ${CBLAS_INCLUDE_DIR}/../lib/mic REALPATH)
    else ()
        get_filename_component(MKL_LIBDIR ${CBLAS_INCLUDE_DIR}/../lib/intel64 REALPATH)
    endif()

    set(LIBMKL_CORE LIBMKL_CORE-NOTFOUND)
    set(LIBMKL_X_THREAD LIBMKL_X_THREAD-NOTFOUND)
    set(LIBMKL_INTEL_XLP64 LIBMKL_INTEL_XLP64-NOTFOUND)

    find_library(LIBMKL_CORE mkl_core ${MKL_LIBDIR})

    if (GHOST_IDX64_LOCAL)
        add_definitions(-DMKL_ILP64)
        find_library(LIBMKL_INTEL_XLP64 mkl_intel_ilp64 ${MKL_LIBDIR})
    else()
        find_library(LIBMKL_INTEL_XLP64 mkl_intel_lp64 ${MKL_LIBDIR})
    endif()

    if(GHOST_USE_OPENMP)
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

    set (BLAS_LIBRARIES "${LIBMKL_INTEL_XLP64};${LIBMKL_CORE};${LIBMKL_X_THREAD}")
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

if (NOT EXISTS "${HWLOC_INCLUDE_DIR}/hwloc.h")
    message(FATAL_ERROR "Could not find hwloc.h neither via pkg-config nor in the defined HWLOC_INCLUDEDIR!")
endif()

include_directories(BEFORE ${HWLOC_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/ghost ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel") 
    if (GHOST_USE_SPMP)
        set(GHOST_HAVE_SPMP 1)
        set(GITHUB_GIT_URL "github.com" CACHE STRING "The github GIT URL. This has to be changed in certain situations if located behind a firewall.")
        include(ExternalProject)
        set(SPMP_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        set_property(DIRECTORY PROPERTY EP_STEP_TARGETS build)
        ExternalProject_Add(SpMP
            PREFIX ${PROJECT_BINARY_DIR}/SpMP
            GIT_REPOSITORY http://${GITHUB_GIT_URL}/IntelLabs/SpMP.git
            CONFIGURE_COMMAND ${PROJECT_SOURCE_DIR}/cmake/configure_spmp.sh ${PROJECT_BINARY_DIR}/SpMP/src/SpMP/synk/loadimba.cpp
            BUILD_COMMAND make CC=${CMAKE_CXX_COMPILER} CCFLAGS=${SPMP_CXX_FLAGS} 
            BUILD_IN_SOURCE 1
            INSTALL_COMMAND ""
            )
        target_link_libraries(ghost "${PROJECT_BINARY_DIR}/SpMP/src/SpMP/libspmp.a")
        include_directories(SYSTEM ${PROJECT_BINARY_DIR}/SpMP/src)
        add_dependencies(ghost SpMP)
    endif()
else()
    message(FATAL_ERROR "SpMP will not build with a compiler which is not Clang/GNU/Intel. Please disable SpMP or use a different compiler!")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
    # msg 193 complains about commented out lines in config.h
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -h c99 -h msglevel_3 -h nomessage=193")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -h msglevel_3 -h nomessage=193")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -c99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++0x")
else ()

########################################################################
# Try to find out, if compiler is Fujitsu
########################################################################
    include(CheckCSourceCompiles)
    unset(CHECK_FUJITSU_DEFINES_C CACHE)
    CHECK_C_SOURCE_COMPILES("
        #if defined(__FUJITSU) || defined(__FCC_VERSION) || defined(__fcc_version)
        int main(){ return 0; }
        #endif
    " CHECK_FUJITSU_DEFINES_C)

    if(${CHECK_FUJITSU_DEFINES_C})
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGHOST_FUJITSU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGHOST_FUJITSU")
    endif()
########################################################################

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -std=c99 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++0x -Wall -Wextra")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_INCLUDE_SYSTEM_FLAG_C "-isystem ") # Not present in Cmake 2.8.11.2
        set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ") # Not present in Cmake 2.8.11.2
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-variable -save-temps -fsource-asm -w3 -wd981")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable -save-temps -fsource-asm -w3 -wd981 -wd383")
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-alias")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-alias")
        endif()
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") 
            message(STATUS "NOTE: by default, we only set the -save-temps compiler flag with Intel compilers right now. You may want to use CC=\"ccache gcc\" etc to speed up the compiliation, see issue #269.")
            if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -fstrict-aliasing")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fstrict-aliasing")
            endif()
            if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND NOT "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "4.8"))
                set(SANITIZER "" CACHE STRING "What should be sanitized by Clang/GCC for a build? (address, thread, memory, undefined or leave empty)")
                if (NOT "${SANITIZER}" STREQUAL "") 
                    string(TOLOWER "${SANITIZER}" SANITIZER)
                    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=${SANITIZER} -fno-omit-frame-pointer -pie")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=${SANITIZER} -fno-omit-frame-pointer -pie")
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

if (GHOST_BUILD_MIC)
    set(GHOST_BUILD_MIC 1 CACHE BOOL "Generate MIC kernels")
    set(GHOST_BUILD_AVX2 0)
    set(GHOST_BUILD_AVX 0)
    set(GHOST_BUILD_SSE 0)
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        message(FATAL_ERROR "The ${CMAKE_CXX_COMPILER_ID} compiler cannot build MIC code. Specify a suited compiler or set GHOST_BUILD_MIC=OFF!")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
else()
    set(GHOST_BUILD_MIC 0 CACHE BOOL "Generate MIC kernels")
    if (GHOST_OPTIMIZE_HOSTARCH)
        AutodetectHostArchitecture()
        if (USE_AVX2)
            set(GHOST_BUILD_AVX2 1 CACHE BOOL "Generate AVX2 kernels")
        else ()
            set(GHOST_BUILD_AVX2 0 CACHE BOOL "Generate AVX2 kernels")
        endif()
        if (USE_AVX)
            set(GHOST_BUILD_AVX 1 CACHE BOOL "Generate AVX kernels")
        else ()
            set(GHOST_BUILD_AVX 0 CACHE BOOL "Generate AVX kernels")
        endif()
        if (USE_SSE2 OR USE_SSE3 OR USE_SSE4_1 OR USE_SSE4_2 OR USE_SSE4a)
            set(GHOST_BUILD_SSE 1 CACHE BOOL "Generate SSE kernels")
        else ()
            set(GHOST_BUILD_SSE 0 CACHE BOOL "Generate SSE kernels")
        endif()

        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -xHost")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")
        endif()
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

if (${BUILD_SHARED_LIBS})
    install(TARGETS ghost EXPORT ghost-targets LIBRARY DESTINATION lib/ghost)
else ()
    install(TARGETS ghost EXPORT ghost-targets ARCHIVE DESTINATION lib/ghost)
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

    
file(GLOB TESTS "${PROJECT_SOURCE_DIR}/test/[^.]*")

find_package(Doxygen)
if (DOXYGEN_FOUND)
    # this pseudo configure_file is needed in order to re-gernated mainpage.md if minimal.c has changed
    configure_file(${PROJECT_SOURCE_DIR}/test/minimal.c ${PROJECT_BINARY_DIR}/test/minimal.c)
    file(READ ${PROJECT_BINARY_DIR}/test/minimal.c GHOST_MINIMAL_EXAMPLE)
    file(READ ${PROJECT_BINARY_DIR}/ghost/config.h GHOST_CONFIG_H)
    configure_file(${PROJECT_SOURCE_DIR}/doxygen/mainpage.md.in ${PROJECT_BINARY_DIR}/doxygen/mainpage.md @ONLY)
    configure_file(${PROJECT_SOURCE_DIR}/doxygen/Doxyfile.in ${PROJECT_BINARY_DIR}/doxygen/Doxyfile @ONLY)
    set(DOC_INSTALL_PREFIX "" CACHE PATH "Where to store the generated documentation")
    mark_as_advanced(DOC_INSTALL_PREFIX)
    file(GLOB DOCFILES "${PROJECT_SOURCE_DIR}/doxygen/[^.]*")
    if (EXISTS ${DOC_INSTALL_PREFIX})
        add_custom_command(
            OUTPUT ${PROJECT_BINARY_DIR}/doc/html/index.html 
            DEPENDS ${SOURCES} ${HEADERS} ${DOCFILES} ${TESTS}
            COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doxygen/Doxyfile && 
            cp ${PROJECT_SOURCE_DIR}/contrib/ghost_logo.png ${PROJECT_BINARY_DIR}/doc/html && 
            rsync -a ${PROJECT_BINARY_DIR}/doc/html/. ${DOC_INSTALL_PREFIX} && 
            find ${DOC_INSTALL_PREFIX} -type d -exec chmod 755 {} + &&
            find ${DOC_INSTALL_PREFIX} -type f -exec chmod 644 {} +
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
            )
    else ()
        add_custom_command(
            OUTPUT ${PROJECT_BINARY_DIR}/doc/html/index.html 
            DEPENDS ${SOURCES} ${HEADERS} ${DOCFILES} ${TESTS}
            COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doxygen/Doxyfile && 
            cp ${PROJECT_SOURCE_DIR}/contrib/ghost_logo.png ${PROJECT_BINARY_DIR}/doc/html 
            )
    endif()

    add_custom_target(doc DEPENDS ${PROJECT_BINARY_DIR}/doc/html/index.html)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/test)
enable_testing()
add_test(NAME simple WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test COMMAND simple)
add_test(NAME minimal WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test COMMAND minimal)
set_tests_properties(minimal PROPERTIES TIMEOUT 60)

# run tests on GPU if CUDA
if (GHOST_USE_CUDA)
  add_test(NAME simple_GPU WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test COMMAND simple)
  set_tests_properties(simple_GPU PROPERTIES ENVIRONMENT GHOST_TYPE=cuda)
  set_tests_properties(simple_GPU PROPERTIES TIMEOUT 60)
  add_test(NAME minimal_GPU WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test COMMAND minimal)
  set_tests_properties(minimal_GPU PROPERTIES ENVIRONMENT GHOST_TYPE=cuda)
  set_tests_properties(minimal_GPU PROPERTIES TIMEOUT 60)
endif()

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --output-on-failure DEPENDS simple)
