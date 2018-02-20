cmake_minimum_required (VERSION 2.8)
project (GHOST)

set(GHOST_VERSION_MAJOR 1)
set(GHOST_VERSION_MINOR 1)
set(GHOST_VERSION_REVISION 2)
set(GHOST_VERSION_IDENTIFIER)
set(GHOST_VERSION "${GHOST_VERSION_MAJOR}.${GHOST_VERSION_MINOR}.${GHOST_VERSION_REVISION}")
if (DEFINED GHOST_VERSION_IDENTIFIER)
    set(GHOST_VERSION "${GHOST_VERSION}-${GHOST_VERSION_IDENTIFIER}")
endif()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

include(CheckFunctionExists)
include(CheckCCompilerFlag)
include(ExternalProject)
include(${PROJECT_SOURCE_DIR}/cmake/OptimizeForArchitecture.cmake)

option(GHOST_USE_CCACHE "Enable/disable using ccache to speed up repeated builds if you use the GNU compiler and ccache is found." ON)
option(GHOST_BUILD_TEST "Build tests (simple, minimal, context, diagfirst" ON)

set(GHOST_USE_MPI 1 CACHE BOOL "Build with MPI")
set(GHOST_USE_SCOTCH 0 CACHE BOOL "Use PT-Scotch for matrix re-ordering")
set(GHOST_USE_COLPACK 0 CACHE BOOL "Use ColPack for matrix coloring")
set(GHOST_USE_ZOLTAN 0 CACHE BOOL "Use Zoltan for matrix re-ordering")
set(GHOST_USE_SPMP 0 CACHE BOOL "Use SpMP for matrix preprocessing")
set(GHOST_USE_RACE 0 CACHE BOOL "Use RACE for recursive matrix coloring")
set(GHOST_USE_GPUDIRECT 0 CACHE BOOL "Use GPUdirect for MPI+CUDA (ignored if CUDA is not enabled)")
set(GHOST_MIC_CROSSCOMPILE 0 CACHE BOOL "Cross-compile for Intel MIC")
set(GHOST_IDX64_LOCAL 0 CACHE BOOL "Enable 64-bit indices for local quantities")
set(GHOST_IDX64_GLOBAL 1 CACHE BOOL "Enable 64-bit indices for global quantities")
set(GHOST_OPTIMIZE_HOSTARCH 1 CACHE BOOL "Create optimized code for the host's architecture")
set(GHOST_LOG_ONLYFIRST 1 CACHE BOOL "Print all errors, warnings and info only once for each occurence")
set(GHOST_LOG_RANK "-1" CACHE STRING "The MPI rank for which the log messages will be printed. -1 for full verbosity.")
set(GHOST_LOG_TIMESTAMP 0 CACHE BOOL "Print a timestamp with each log message.")
set(GHOST_TRACK_DATATRANSFERS 0 CACHE BOOL "Track MPI and CUDA data transfers")
set(GHOST_SPARSEMAT_GLOBALSTATS 0 CACHE BOOL "Gather some global statistics on sparse matrices (should be disabled for large-scale runs)")
set(GHOST_SPARSEMAT_STATS 0 CACHE BOOL "Gather statistics about sparse matrices (slows down matrix construction)")
set(GHOST_KACZ_ANALYZE 0 CACHE BOOL "Gather data about load balancing for KACZ method (might slow down initial splitting)")  
set(GHOST_AUTOGEN_SPMMV "8,1;8,2;8,4;8,8;8,*;32,1;32,2;32,4;32,8;32,*;1,4;1,*" CACHE STRING "SpM(M)V kernels to be auto-generated. Semicolon-separated list of 2-tuples: (CHUNKHEIGHT, NVECS)")
set(GHOST_AUTOGEN_GS "1,1" CACHE STRING "GS kernels to be auto-generated. Semicolon-separated list of 2-tuples: (CHUNKHEIGHT, NVECS)")
set(GHOST_AUTOGEN_SPMMV_CUDA "32,1,0,0,0,0,0,0,0;1,4,0,0,0,0,0,0,0;32,*,0,0,0,0,0,0,0;1,*,0,0,0,0,0,0,0" CACHE STRING "SpM(M)V kernels to be auto-generated. Semicolon-separated list of 9-tuples: (CHUNKHEIGHT, NVECS,AXP(B)Y,SCALE,(V)SHIFT,DOT_YY,DOT_XY,DOT_XX,CHAIN_AXPBY)")
set(GHOST_AUTOGEN_TSMM "1,1;2,2;4,4;8,8" CACHE STRING "TSMM kernels to be auto-generated. Semicolon-separated list of 3-tuples: (K, N, UNROLL)")
set(GHOST_AUTOGEN_TSMM-INPLACE "1,1;2,2;4,4;8,8" CACHE STRING "TSMM-inplace kernels to be auto-generated. Semicolon-separated list of 2-tuples: (K, N) where K>=N")
set(GHOST_AUTOGEN_TSMTTSM "1,1;2,2;3,3;4,4;5,5;6,6;7,7;8,8;16,16;32,32;1,*;2,*;3,*;4,*;5,*;6,*;7,*;8,*;9,*,10,*;11,*;12,*;13,*;14,*;15,*;16,*;*,1;*,2;*,3;*,4;*,5;*,6;*,7;*,8;*,9;*,10;*,11;*,12;*,13;*,14;*,15;*,16;" CACHE STRING "TSMTTSM kernels to be auto-generated. Semicolon-separated list of 3-tuples: (M, N, UNROLL)")
set(GHOST_AUTOGEN_DOT "1;2;4;8" CACHE STRING "DOT kernels to be auto-generated. Semicolon-separated list of 1-tuples: (NVECS)")
set(GHOST_AUTOGEN_KACZ "32,1,0;32,4,0;1,4,0;1,4,4" CACHE STRING "KACZ kernels to be auto-generated. Semicolon-separated list of 3-tuples: (CHUNKHEIGHT, NVECS, NSHIFTS)")
set(GHOST_VERBOSITY 1 CACHE STRING "Level of verbosity information. Valid values: 0 (be totally silent), 1 (print warnings, errors and further important information), 2, 3, 4 (increasing level of verbosity)")
set(GHOST_INSTRUMENT "" CACHE STRING "Type of instrumentation (semicolon separated list). Valid values: TIMING, LIKWID, IACA")
set(GHOST_INSTRUMENT_FUNCTYPES "(functype) & GHOST_FUNCTYPE_ANY" CACHE STRING "Logical selection of function types to instrument. Possible values are defined in func_util.h and functype should be put in parentheses.")
set(GHOST_STREAM_ARRAY_SIZE "1e8" CACHE STRING "Array size used for STREAM micro-benchmarking. Adjust of memory is scarce.")
option(GHOST_INSTRUMENT_BARRIER "Place a barrier after each instrumented function (may be necessary for correct timings)" 0)
option(GHOST_COMPATIBLE_CHECK "Check for compatible data in kernels" 1)
option(GHOST_COMPATIBLE_PERM "Allows GHOST to permute and assign maps for compatibility. Set to ON if you are not sure about the permutations." 0)  

set(GHOST_AUTOGEN_SPMMV_BASE "*,*" CACHE INTERNAL "SpM(M)V kernels to be always auto-generated. Semicolon-separated list of 2-tuples: (CHUNKHEIGHT, NVECS)")
set(GHOST_AUTOGEN_TSMM_BASE "*,*" CACHE INTERNAL "TSMM kernels to be auto-generated. Semicolon-separated list of 3-tuples: (K, N, UNROLL)")
set(GHOST_AUTOGEN_TSMM-INPLACE_BASE "*,*" CACHE INTERNAL "TSMM-inplace kernels to be auto-generated. Semicolon-separated list of 2-tuples: (K, N)")
set(GHOST_AUTOGEN_TSMTTSM_BASE "*,*" CACHE INTERNAL "TSMTTSM kernels to be auto-generated. Semicolon-separated list of 3-tuples: (M, N, UNROLL)")
set(GHOST_AUTOGEN_DOT_BASE "*" CACHE INTERNAL "DOT kernels to be auto-generated. Semicolon-separated list of 1-tuples: (NVECS)")
set(GHOST_AUTOGEN_KACZ_BASE "*,*,*" CACHE INTERNAL "KACZ kernels to be auto-generated. Semicolon-separated list of 3-tuples: (CHUNKHEIGHT, NVECS, NSHIFTS)")

set(GHOST_AUTOGEN_SPMMV ${GHOST_AUTOGEN_SPMMV_BASE} ${GHOST_AUTOGEN_SPMMV})
set(GHOST_AUTOGEN_TSMM ${GHOST_AUTOGEN_TSMM_BASE} ${GHOST_AUTOGEN_TSMM})
set(GHOST_AUTOGEN_TSMM-INPLACE ${GHOST_AUTOGEN_TSMM-INPLACE_BASE} ${GHOST_AUTOGEN_TSMM-INPLACE})
set(GHOST_AUTOGEN_TSMTTSM ${GHOST_AUTOGEN_TSMTTSM_BASE} ${GHOST_AUTOGEN_TSMTTSM})
set(GHOST_AUTOGEN_DOT ${GHOST_AUTOGEN_DOT_BASE} ${GHOST_AUTOGEN_DOT})
set(GHOST_AUTOGEN_KACZ ${GHOST_AUTOGEN_KACZ_BASE} ${GHOST_AUTOGEN_KACZ})

set(GHOST_AUTOGEN_MAX_CHUNKHEIGHT 1)
foreach(SPMMVCONF ${GHOST_AUTOGEN_SPMMV})
    string(REPLACE "," ";" SPMMVCONF_LIST ${SPMMVCONF})
    list(GET SPMMVCONF_LIST 0 CHUNKHEIGHT)
    if(NOT "${CHUNKHEIGHT}" STREQUAL "*" AND ${CHUNKHEIGHT} GREATER ${GHOST_AUTOGEN_MAX_CHUNKHEIGHT})
        set(GHOST_AUTOGEN_MAX_CHUNKHEIGHT ${CHUNKHEIGHT})
    endif()
endforeach()

find_package(PkgConfig)

add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX)
if (GHOST_IDX64_LOCAL)
    if (NOT GHOST_IDX64_GLOBAL)
        message(FATAL_ERROR "Having 64-bit indices locally and only 32-bit indices globally is not allowed (and not sensible)!")
    endif()
endif()

if (GHOST_COMPATIBLE_PERM AND NOT GHOST_COMPATIBLE_CHECK)
    message(FATAL_ERROR "GHOST_COMPATIBLE_PERM can only be enabled together with GHOST_COMPATIBLE_CHECK!")
endif()

set (SOURCES
    src/autogen.cpp
    src/datatransfers.cpp 
    src/densemat.cpp
    src/densemat_cm_averagehalo.cpp
    src/densemat_cm.cpp 
    src/densemat_rm_averagehalo.cpp
    src/densemat_rm.cpp
    src/dot.cpp 
    src/errorhandler.cpp
    src/rcm_dissection.cpp
    src/sell_kacz_fallback.cpp 
    src/sell_kacz.cpp
    src/sell_spmv.cpp
    src/sparsemat.cpp
    src/sparsemat_perm_blockColor.cpp 
    src/sparsemat_perm_colpack.cpp
    src/sparsemat_perm_spmp.cpp
    src/timing.cpp
    src/tsmm.cpp
    src/tsmm_inplace.cpp
    src/tsmttsm.cpp
    src/tsmtspmtsm.cpp
    src/tsmtspmtsm_var2_cuda.cu
    src/tsmtspmtsm_var2_plain.cpp
    src/bench.c
    src/bincrs_func.c
    src/bitmap.c
    src/blas_util.c
    src/carp.c
    src/carp_rb.c
    src/compatibility_check.c
    src/context.c
    src/context.cpp
    src/core.c
    src/cu_util.c
    src/cu_temp_buffer_malloc.cpp
    src/densemat.c
    src/densemat_cm.c 
    src/densemat_rm.c 
    src/error.c
    src/gemm.c
    src/instr.c
    src/kacz_analyze_print.c
    src/kacz_hybrid_split.c
    src/locality.c
    src/machine.c
    src/map.c
    src/math.c
    src/matrixmarket.c
    src/mmio.c
    src/omp.c
    src/perm.c
    src/pumap.c
    src/rand.c
    src/sell_kacz.c
    src/sell_kacz_rb.c
    src/sell_kacz_rb_versions.c 
    src/sell_kacz_bmc.c 
    src/sell_kacz_mc.c
    src/sparsemat.c
    src/sparsemat_perm_scotch.c
    src/sparsemat_perm_sort.c
    src/sparsemat_perm_zoltan.c
    src/spmv.c
    src/spmv_overlap.c 
    src/spmv_pipelined.c
    src/spmv_task.c
    src/spmv_vector.c 
    src/task.c
    src/taskq.c
    src/thpool.c
    src/timing.c
    src/types.c
    src/util.c
    src/sparsemat_perm_RACE.cpp
    src/sell_gs_RACE_fallback.cpp
    src/sell_gs.cpp
    )

set (HEADERS
    include/ghost/autogen.h
    include/ghost/bench.h
    include/ghost/bincrs.h
    include/ghost/bindensemat.h
    include/ghost/bitmap.h
    include/ghost/blas_mangle.h
    include/ghost/carp.h
    include/ghost/context.h
    include/ghost/core.h
    include/ghost/cu_bench.h
    include/ghost/cu_util.h
    include/ghost/cu_temp_buffer_malloc.h
    include/ghost/datatransfers.h
    include/ghost/densemat.h
    include/ghost/densemat_cm.h
    include/ghost/densemat_iter_macros.h
    include/ghost/densemat_rm.h
    include/ghost/dot.h
    include/ghost/error.h
    include/ghost/errorhandler.h
    include/ghost/helper.h
    include/ghost/instr.h
    include/ghost/locality.h	
    include/ghost/log.h
    include/ghost/machine.h
    include/ghost/map.h
    include/ghost/math.h
    include/ghost/matrixmarket.h
    include/ghost/omp.h
    include/ghost/pumap.h
    include/ghost/rand.h
    include/ghost/sparsemat.h
    include/ghost/sparsemat_src.h
    include/ghost/spmv.h
    include/ghost/spmv_solvers.h
    include/ghost/task.h
    include/ghost/taskq.h
    include/ghost/thpool.h
    include/ghost/timing.h
    include/ghost/tsmm.h
    include/ghost/tsmm_inplace.h
    include/ghost/tsmttsm.h
    include/ghost/tsmtspmtsm.h
    include/ghost/types.h
    include/ghost/util.h
    include/ghost/rcm_dissection.h
    include/ghost/sell_kacz_rb.h
    include/ghost/carp_rb.h
    include/ghost/sell_kacz_rb_versions.h 
    include/ghost/kacz_hybrid_split.h
    include/ghost/sell_kacz_bmc.h 
    include/ghost/sell_kacz_mc.h
    include/ghost/sell_kacz_fallback.h
    include/ghost/compatibility_check.h
    )

# only do this in the initial run: either if GHOST_USE_CUDA is undefined or it is enabled by the user but no libraries are found yet
if (NOT DEFINED GHOST_USE_CUDA OR (GHOST_USE_CUDA AND ("${CUDA_LIBRARIES}" STREQUAL "")))
    option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    find_package(CUDA)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE) # unfortunately, this gets overwritten in FindCUDA.cmake, so we have to force set it again
    find_package(CUDA)
endif()

if (CUDA_FOUND OR GHOST_USE_CUDA)
    find_package(CUDA)
    set(GHOST_USE_CUDA 1 CACHE BOOL "Build with CUDA")
    set(GHOST_HAVE_CUDA 1)
else ()
    set(GHOST_USE_CUDA 0 CACHE BOOL "Build with CUDA")
    set(CUDA_LIBRARIES "")
endif()

set (PP_SCRIPT ${PROJECT_SOURCE_DIR}/bin/ghost_unroll.pl)
set (PP_HEADER_SCRIPT ${PROJECT_SOURCE_DIR}/bin/ghost_extractfunc.pl)
set (PP_KERNELMAP_SCRIPT ${PROJECT_SOURCE_DIR}/bin/ghost_mapfunc.pl)
set (UNROLLED_SRC_DIR ${PROJECT_BINARY_DIR}/src_unrolled/)
set (GENERATED_INC_DIR ${PROJECT_BINARY_DIR}/ghost/)

# stolen from http://stackoverflow.com/questions/12521452
function (getListOfVarsStartingWith _prefix _varResult)
    get_cmake_property(_vars CACHE_VARIABLES)
    string (REGEX MATCHALL "(^|;)${_prefix}[A-Za-z0-9_-]*" _matchedVars "${_vars}")
    set (_resultVars "")
    foreach (_variable ${_matchedVars})
        get_property(_type CACHE "${_variable}" PROPERTY TYPE)
        if (NOT "${_type}" STREQUAL "STATIC") 
            list (APPEND _resultVars "${_variable}")
        endif()
    endforeach()
    set (${_varResult} ${_resultVars} PARENT_SCOPE)
endfunction()

getListOfVarsStartingWith(GHOST_AUTOGEN GHOST_AUTOGEN_FUNC_LIST)

if (GHOST_HAVE_CUDA)
    file(GLOB TEMPLATE_SRCFILES "${CMAKE_SOURCE_DIR}/src/autogen/*.in")
else()
    file(GLOB TEMPLATE_SRCFILES "${CMAKE_SOURCE_DIR}/src/autogen/*.c.in" "${CMAKE_SOURCE_DIR}/src/autogen/*.cpp.in")
endif()

foreach(TEMPLATE_SRCFILE ${TEMPLATE_SRCFILES})
    set(GENERATED_FILES "")
    set(UNROLLED_FILES "")
    get_filename_component(TEMPLATE_SRCFILE_BASE ${TEMPLATE_SRCFILE} NAME_WE)
    get_filename_component(TEMPLATE_SRCFILE_FULLEXT ${TEMPLATE_SRCFILE} EXT)
    string(REPLACE ".in" "" TEMPLATE_SRCFILE_EXT ${TEMPLATE_SRCFILE_FULLEXT})

    file(STRINGS ${TEMPLATE_SRCFILE} TEMPLATE_HEADER LIMIT_COUNT 1)
    string(STRIP ${TEMPLATE_HEADER} TEMPLATE_HEADER)
    if (NOT "${TEMPLATE_HEADER}" STRGREATER "/*!GHOST_AUTOGEN_")
        #        message(FATAL_ERROR "AUTOGEN header missing or wrong in file ${TEMPLATE_SRCFILE}. Is ${TEMPLATE_HEADER}, should be /*!GHOST_AUTOGEN_*")
    else ()
        foreach (GHOST_AUTOGEN_FUNC ${GHOST_AUTOGEN_FUNC_LIST})
            set(FUNC_HEADER "/*!${GHOST_AUTOGEN_FUNC}")
            foreach (COMBINATION ${${GHOST_AUTOGEN_FUNC}})
                if ("${TEMPLATE_HEADER} " MATCHES "${FUNC_HEADER} ") # File matches AUTOGEN function
                    string(REGEX MATCH " .* " TEMPLATE_CONF ${TEMPLATE_HEADER}) # match string between spaces which contains the conf
                    string(STRIP ${TEMPLATE_CONF} TEMPLATE_CONF)

                    # check if fixed parameters match: replace any non-fixed parameters with ".+" and match this against the combination
                    # every non-fixed will now match _any_ number and fixed numbers have to match the specific number
                    string(REGEX REPLACE "[a-zA-Z\\*]+" ".+" TEMPLATE_CONF_FIXEDPARS ${TEMPLATE_CONF})
                    string(REGEX MATCH ${TEMPLATE_CONF_FIXEDPARS} FIXEDMATCH ${COMBINATION})

                    # check if wildcards match: replace any non-wildcards with "." and compare strings
                    string(REGEX REPLACE "[a-zA-Z0-9]+" "." TEMPLATE_CONF_PATTERN ${TEMPLATE_CONF})
                    string(REGEX REPLACE "[a-zA-Z0-9]+" "." COMBINATION_PATTERN ${COMBINATION})


                    if (NOT ${FIXEDMATCH} STREQUAL "" AND
                            ${TEMPLATE_CONF_PATTERN} STREQUAL ${COMBINATION_PATTERN})
                        string(REPLACE "," ";" TEMPLATE_CONF_LIST ${TEMPLATE_CONF})
                        string(REPLACE "," ";" COMBINATION_LIST ${COMBINATION})
                        list(LENGTH TEMPLATE_CONF_LIST numPars)
                        math(EXPR numPars_ "${numPars} - 1")
                        foreach(parIdx RANGE ${numPars_})
                            list(GET TEMPLATE_CONF_LIST ${parIdx} PARAMETER)
                            list(GET COMBINATION_LIST ${parIdx} VALUE)
                            set(${PARAMETER} ${VALUE}) # set a variable called PARAMETER to VALUE
                        endforeach()

                        string(REGEX REPLACE "\\*" "var" COMBINATION_FILENAME ${COMBINATION})
                        string(REGEX REPLACE "," "_" COMBINATION_FILENAME ${COMBINATION_FILENAME})
                        set(GENERATED_FILE ${PROJECT_BINARY_DIR}/src_generated/${TEMPLATE_SRCFILE_BASE}__${COMBINATION_FILENAME}${TEMPLATE_SRCFILE_EXT})
                        configure_file (
                            ${TEMPLATE_SRCFILE}
                            ${GENERATED_FILE}
                            )
                        list(APPEND GENERATED_FILES ${GENERATED_FILE})
                    endif()
                endif()
            endforeach()
        endforeach()
    endif()

    foreach (GENERATED_FILE ${GENERATED_FILES})
        #message(STATUS "Unrolling ${GENERATED_FILE}")
        get_filename_component(TO_UNROLL_FILENAME ${GENERATED_FILE} NAME)
        get_filename_component(TO_UNROLL_FILENAME_WE ${GENERATED_FILE} NAME_WE)
        set(UNROLLED_FILE ${UNROLLED_SRC_DIR}/${TO_UNROLL_FILENAME})
        add_custom_command(
            OUTPUT ${UNROLLED_FILE}
            DEPENDS ${GENERATED_FILE} ${PP_SCRIPT} 
            COMMAND mkdir -p ${UNROLLED_SRC_DIR} && perl ${PP_SCRIPT} ${GENERATED_FILE} > ${UNROLLED_FILE}
            )
        list(APPEND UNROLLED_FILES ${UNROLLED_FILE})
    endforeach()

    set(GENERATED_HEADER ${GENERATED_INC_DIR}/${TEMPLATE_SRCFILE_BASE}_gen.h)
    set(GENERATED_KERNELMAP ${GENERATED_INC_DIR}/${TEMPLATE_SRCFILE_BASE}.def)
    #message(STATUS "${TEMPLATE_SRCFILE_BASE} ${GENERATED_HEADER} ${GENERATED_KERNELMAP}")
    #message(STATUS "GEN header ${GENERATED_HEADER} from file ${UNROLLED_FILES}")
    add_custom_command(
        OUTPUT ${GENERATED_HEADER}
        DEPENDS ${UNROLLED_FILES} ${PP_HEADER_SCRIPT} 
        COMMAND mkdir -p ${GENERATED_INC_DIR} && perl ${PP_HEADER_SCRIPT} ${UNROLLED_FILES} > ${GENERATED_HEADER}
        )
    list(APPEND HEADERS ${GENERATED_HEADER})
    add_custom_command(
        OUTPUT ${GENERATED_KERNELMAP}
        DEPENDS ${GENERATED_HEADER} ${PP_KERNELMAP_SCRIPT} 
        COMMAND mkdir -p ${GENERATED_INC_DIR} && perl ${PP_KERNELMAP_SCRIPT} ${GENERATED_HEADER} > ${GENERATED_KERNELMAP}
        )
    if (${TEMPLATE_SRCFILE_EXT} STREQUAL ".cu")
        list(APPEND CU_GENERATED_SOURCES ${UNROLLED_FILES})
    else()
        list(APPEND SOURCES ${UNROLLED_FILES})
    endif()
    list(APPEND GLOB_UNROLLED_FILES ${UNROLLED_FILE})
    list(APPEND GLOB_GENERATED_FILES ${GENERATED_FILE})
    list(APPEND GLOB_GENERATED_HEADERS ${GENERATED_HEADER})
    list(APPEND GLOB_GENERATED_KERNELMAPS ${GENERATED_KERNELMAP})
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
    find_package(HWLOC)
    find_library(LIBHWLOC hwloc)
endif()

# find hwloc.h in default header locations
if (NOT HWLOC_INCLUDE_DIR)
    find_path(HWLOC_INCLUDE_DIR hwloc.h)
endif()
        
set(GITHUB_GIT_URL "github.com" CACHE STRING "The github GIT URL. This has to be changed in certain situations if located behind a firewall.")

# hwloc not found - download and build it
if (NOT HWLOC_INCLUDE_DIR)
    message(STATUS "Could not find hwloc.h neither via pkg-config nor in the defined HWLOC_INCLUDE_DIR! Trying to build hwloc...")
    ExternalProject_Add(hwloc
        PREFIX ${PROJECT_BINARY_DIR}/hwloc
        URL https://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-1.11.6.tar.gz
        SOURCE_DIR ${PROJECT_BINARY_DIR}/hwloc/src/hwloc
        CONFIGURE_COMMAND ./configure --prefix=${PROJECT_BINARY_DIR}/hwloc-inst
        BUILD_COMMAND make 
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND make install
        )
    set(BUILTIN_HWLOC 1)
    set(HWLOC_INCLUDE_DIR "${PROJECT_BINARY_DIR}/hwloc-inst/include")
    # Note that on some systems it is lib64 and we might also be static!
    set(LIBHWLOC "${PROJECT_BINARY_DIR}/hwloc-inst/lib/libhwloc.so")
endif()

# HWLOC_LIBRARIES might have been set by the user
if(NOT HWLOC_LIBRARIES)
  if(NOT LIBHWLOC)
    get_filename_component(HWLOC_LIBDIR ${HWLOC_INCLUDE_DIR}/../lib REALPATH)
    find_library(LIBHWLOC hwloc HINTS ${HWLOC_LIBDIR})
  endif()
  set(HWLOC_LIBRARIES ${LIBHWLOC})
endif()   

if (NOT EXISTS "${HWLOC_INCLUDE_DIR}/hwloc.h")
endif()

include_directories(BEFORE ${HWLOC_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/ghost ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include)


find_package(OpenMP)

if (OPENMP_FOUND)	
    set(GHOST_USE_OPENMP 1 CACHE BOOL "Build with OpenMP")
    set(GHOST_HAVE_OPENMP 1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
    set(GHOST_USE_OPENMP 0 CACHE BOOL "Build with OpenMP")
    set(GHOST_HAVE_OPENMP 0)
endif()


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
        if (GHOST_MIC_CROSSCOMPILE)
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
    set(ZOLTAN_LIBRARIES ${ZOLTAN_LIBRARIES} -lm)
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

if (GHOST_USE_RACE)
    set(GHOST_HAVE_RACE 1)
    set(RACE_INCLUDE_DIR "${RACE_INCLUDE_DIR}" CACHE PATH "Location of RACE headers")
    include_directories(${RACE_INCLUDE_DIR})
    get_filename_component(RACE_LIBDIR ${RACE_INCLUDE_DIR}/../lib/RACE REALPATH)

    find_library(RACE_LIBRARIES NAMES RACE PATHS ${RACE_LIBDIR})
    if (NOT RACE_LIBRARIES)
        message(FATAL_ERROR "RACE_INCLUDE_DIR not set correctly!")
    endif()
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

find_file(CBLAS_H_FULL NAMES "mkl_cblas.h" "cblas.h"  "gsl_cblas.h" CBLAS_H_FULL PATHS ${CBLAS_INCLUDE_DIR} PATH_SUFFIXES "cblas")
get_filename_component(CBLAS_H ${CBLAS_H_FULL} NAME)

if (${CBLAS_H} STREQUAL "mkl_cblas.h")
    message(STATUS "MKL BLAS found")
    set(GHOST_HAVE_MKL 1)
    get_filename_component(MKL_LIBDIR ${CBLAS_INCLUDE_DIR}/../lib/ REALPATH)
    if (GHOST_MIC_CROSSCOMPILE)
        set(MKL_PATH_SUFFIXES "mic")
    else ()
        set(MKL_PATH_SUFFIXES "intel64")
    endif()

    set(LIBMKL_CORE LIBMKL_CORE-NOTFOUND)
    set(LIBMKL_X_THREAD LIBMKL_X_THREAD-NOTFOUND)
    set(LIBMKL_INTEL_XLP64 LIBMKL_INTEL_XLP64-NOTFOUND)

    find_library(LIBMKL_CORE mkl_core PATHS ${MKL_LIBDIR} PATH_SUFFIXES ${MKL_PATH_SUFFIXES})

    if (GHOST_IDX64_LOCAL)
        add_definitions(-DMKL_ILP64)
        find_library(LIBMKL_INTEL_XLP64 mkl_intel_ilp64 PATHS ${MKL_LIBDIR} PATH_SUFFIXES ${MKL_PATH_SUFFIXES})
    else()
        find_library(LIBMKL_INTEL_XLP64 mkl_intel_lp64 PATHS ${MKL_LIBDIR} PATH_SUFFIXES ${MKL_PATH_SUFFIXES})
    endif()

    if(GHOST_USE_OPENMP)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel") 
            find_library(LIBMKL_X_THREAD mkl_intel_thread PATHS ${MKL_LIBDIR} PATH_SUFFIXES ${MKL_PATH_SUFFIXES})
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") 
            find_library(LIBMKL_X_THREAD mkl_gnu_thread PATHS ${MKL_LIBDIR} PATH_SUFFIXES ${MKL_PATH_SUFFIXES})
        else ()
            message(FATAL_ERROR "OpenMP only possible with Intel or Gnu!")
        endif()
    else ()
        find_library(LIBMKL_X_THREAD mkl_sequential PATHS ${MKL_LIBDIR} PATH_SUFFIXES ${MKL_PATH_SUFFIXES})
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


if (GHOST_USE_GPUDIRECT)
    set(GHOST_HAVE_GPUDIRECT 1)
endif()

if (GHOST_USE_SPMP)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(GHOST_HAVE_SPMP 1)
        set(SPMP_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        set_property(DIRECTORY PROPERTY EP_STEP_TARGETS build)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            ExternalProject_Add(SpMP
                PREFIX ${PROJECT_BINARY_DIR}/SpMP
                URL https://github.com/IntelLabs/SpMP/archive/master.zip
                CONFIGURE_COMMAND ""
                BUILD_COMMAND make 
                BUILD_IN_SOURCE 1
                INSTALL_COMMAND ""
                )
        else() # GNU or Clang: Set CUSTOM_CXX variable in SpMP make
            ExternalProject_Add(SpMP
                PREFIX ${PROJECT_BINARY_DIR}/SpMP
                URL https://github.com/IntelLabs/SpMP/archive/master.zip
                CONFIGURE_COMMAND ""
                BUILD_COMMAND make CUSTOM_CXX=${CMAKE_CXX_COMPILER} 
                BUILD_IN_SOURCE 1
                INSTALL_COMMAND ""
                )
        endif()
    else()
        message(FATAL_ERROR "SpMP will not build with a compiler which is not Clang/GNU/Intel. Please disable SpMP or use a different compiler!")
    endif()
endif()

if (GHOST_HAVE_CUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    cuda_include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR} ${PROJECT_BINARY_DIR}/ghost)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC;-Xcompiler;-Wall)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
    if(CUDA_VERSION VERSION_GREATER "8.99.99")
      list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
    endif()
    list(APPEND CUDA_NVCC_FLAGS -isystem;${HWLOC_INCLUDE_DIR})
    foreach(mpi_inc ${MPI_INCLUDE_PATH}) 
        list(APPEND CUDA_NVCC_FLAGS -isystem;${mpi_inc})
    endforeach()
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        list(APPEND CUDA_NVCC_FLAGS -O0;-G;-g)
    endif()
    if (GHOST_USE_OPENMP)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fopenmp)
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

add_library(ghost ${SOURCES})
add_dependencies(ghost autogen)

if (GHOST_USE_SPMP)
    target_link_libraries(ghost "${PROJECT_BINARY_DIR}/SpMP/src/SpMP/libspmp.a")
    include_directories(SYSTEM ${PROJECT_BINARY_DIR}/SpMP/src)
    add_dependencies(ghost SpMP)
endif()

if (BUILTIN_HWLOC)
    target_link_libraries(ghost "${PROJECT_BINARY_DIR}/hwloc-inst/lib/libhwloc.so")
    include_directories(SYSTEM ${PROJECT_BINARY_DIR}/hwloc-inst/include)
    add_dependencies(ghost hwloc)
endif()

if (GHOST_USE_RACE)
    target_link_libraries(ghost "${RACE_LIBRARIES}")
endif()


check_function_exists(qsort_r HAVE_QSORT_R)

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
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGHOST_FUJITSU -noansi")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGHOST_FUJITSU")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -std=c99 -Wall -Wextra -pedantic")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++0x -Wall -Wextra -pedantic")
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

                if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") 

                    if (NOT GHOST_HAVE_CUDA)
                      find_program(CCACHE_FOUND ccache)
                      if(CCACHE_FOUND AND GHOST_USE_CCACHE)
                          set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
                          set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
                      endif()
                    endif()
                endif()
              
                if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
                    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -fstrict-aliasing")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fstrict-aliasing")
                endif()
                if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND NOT "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "4.8"))
                    set(SANITIZER "" CACHE STRING "What should be sanitized by Clang/GCC for a build? (address, thread, memory, undefined or leave empty)")
                    if (NOT "${SANITIZER}" STREQUAL "") 
                        string(TOLOWER "${SANITIZER}" SANITIZER)
                        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=${SANITIZER} -fno-omit-frame-pointer")
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=${SANITIZER} -fno-omit-frame-pointer")
                        if (GHOST_HAVE_CUDA)
                            message(WARNING "Sanitizers are probably not working together with CUDA!")
                        endif()
                    endif()
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

if (GHOST_MIC_CROSSCOMPILE)
    set(GHOST_BUILD_MIC 1 CACHE BOOL "Generate MIC kernels")
    set(GHOST_BUILD_AVX2 0)
    set(GHOST_BUILD_AVX 0)
    set(GHOST_BUILD_SSE 0)
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        message(FATAL_ERROR "The ${CMAKE_CXX_COMPILER_ID} compiler cannot build MIC code. Specify a suited compiler or set GHOST_MIC_CROSSCOMPILE=OFF!")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
else()
    set(GHOST_BUILD_MIC 0 CACHE BOOL "Generate MIC kernels")
    if (GHOST_OPTIMIZE_HOSTARCH)
        AutodetectHostArchitecture()
        OptimizeForArchitecture()
        if (USE_AVX512F)
          if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -xCOMMON-AVX512")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xCOMMON-AVX512")
            set(GHOST_BUILD_MIC 1 CACHE BOOL "Generate MIC kernels" FORCE)
          else ()
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
          endif()
          
          set(GHOST_BUILD_AVX512 1 CACHE BOOL "Generate AVX512 kernels")
          
        else ()
            set(GHOST_BUILD_AVX512 0 CACHE BOOL "Generate AVX512 kernels")
        endif()
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

#        if (NOT USE_AVX512F)
        if (TRUE)
            if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -xHost")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")
            elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
            endif()
        endif()
    endif()
endif()


if( ${GHOST_BUILD_MIC} )
  message(STATUS "Build with MIC")
endif()
if( ${GHOST_BUILD_AVX512} )
  message( STATUS "Build with AVX512")
endif()
if( ${GHOST_BUILD_AVX2} )
  message( STATUS "Build with AVX2")
endif()
if( ${GHOST_BUILD_AVX} )
  message(STATUS "Build with AVX")
endif()
if( ${GHOST_BUILD_SSE} )
  message(STATUS "Build with SSE")
endif()




########################################################################
#  Disable a few warnings for release builds.
########################################################################
#if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
#    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function -Wno-unused-but-set-variable")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function -Wno-unused-but-set-variable")
#    endif()
# TODO Are there any other flags?
#    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
# endif()
#endif()
########################################################################

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
    if (NOT ${DOC_INSTALL_PREFIX} STREQUAL "")
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

if(GHOST_BUILD_TEST)
  add_subdirectory(${PROJECT_SOURCE_DIR}/test)
  enable_testing()
  add_test(NAME simple WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ./simple)
  add_test(NAME minimal WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ./minimal)
  add_test(NAME context WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ./context)
  add_test(NAME diagfirst WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ./diagfirst)
endif()  

find_program(MPIRUN NAMES mpirun mpiexec mpiexec.hydra)


# run tests on GPU if one is present
# code adapted from http://stackoverflow.com/a/37995691
set(DETECT_GPU_COUNT 0)
if (GHOST_USE_CUDA)
    find_program(_nvidia_smi_path "nvidia-smi")
    mark_as_advanced(_nvidia_smi_path)
    if (_nvidia_smi_path)
        set(DETECT_GPU_COUNT_NVIDIA_SMI 0)
        # execute nvidia-smi -L to get a short list of GPUs available
        exec_program(${_nvidia_smi_path} ARGS -L
            OUTPUT_VARIABLE _nvidia_smi_out
            RETURN_VALUE    _nvidia_smi_ret)
        # process the stdout of nvidia-smi
        if (_nvidia_smi_ret EQUAL 0)
            # convert string with newlines to list of strings
            string(REGEX REPLACE "\n" ";" _nvidia_smi_out "${_nvidia_smi_out}")
            foreach(_line ${_nvidia_smi_out})
                if (_line MATCHES "^GPU [0-9]+:")
                    math(EXPR DETECT_GPU_COUNT_NVIDIA_SMI "${DETECT_GPU_COUNT_NVIDIA_SMI}+1")
                    # the UUID is not very useful for the user, remove it
                    string(REGEX REPLACE " \\(UUID:.*\\)" "" _gpu_info "${_line}")
                    if (NOT _gpu_info STREQUAL "")
                        list(APPEND DETECT_GPU_INFO "${_gpu_info}")
                    endif()
                endif()
            endforeach()

            set(DETECT_GPU_COUNT ${DETECT_GPU_COUNT_NVIDIA_SMI})
        endif()
    endif()
endif()

if(GHOST_BUILD_TEST)
  if (${DETECT_GPU_COUNT} GREATER 0)
      add_test(NAME simple_GPU WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ./simple)
      set_tests_properties(simple_GPU PROPERTIES ENVIRONMENT GHOST_TYPE=GPU)
      add_test(NAME minimal_GPU WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ./minimal)
      set_tests_properties(minimal_GPU PROPERTIES ENVIRONMENT GHOST_TYPE=GPU)
    
      add_test(NAME simple_np2 WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ${MPIRUN} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_POSTFLAGS} ./simple)
      add_test(NAME minimal_np2 WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ${MPIRUN} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_POSTFLAGS} ./minimal)
      add_test(NAME context_np2 WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ${MPIRUN} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_POSTFLAGS} ./context)
     add_test(NAME diagfirst_np2 WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ${MPIRUN} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_POSTFLAGS} ./diagfirst)
  else()
      add_test(NAME simple_np2 WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ${MPIRUN} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_POSTFLAGS} ./simple)
      add_test(NAME minimal_np2 WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ${MPIRUN} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_POSTFLAGS} ./minimal)
      add_test(NAME context_np2 WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ${MPIRUN} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_POSTFLAGS} ./context)
      add_test(NAME diagfirst_np2 WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test COMMAND ${MPIRUN} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_POSTFLAGS} ./diagfirst)
    
      set_tests_properties(minimal_np2 PROPERTIES ENVIRONMENT GHOST_TYPE=CPU)
      set_tests_properties(simple_np2 PROPERTIES ENVIRONMENT GHOST_TYPE=CPU)
      set_tests_properties(context_np2 PROPERTIES ENVIRONMENT GHOST_TYPE=CPU)
      set_tests_properties(diagfirst_np2 PROPERTIES ENVIRONMENT GHOST_TYPE=CPU)
  endif()
  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --output-on-failure DEPENDS simple minimal context diagfirst)
endif(GHOST_BUILD_TEST)   


