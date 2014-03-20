cmake_minimum_required (VERSION 2.8)
project (GHOST)

set(GHOST_VERSION_MAJOR 0)
set(GHOST_VERSION_MINOR 7)
set(GHOST_VERSION_REVISION 0)
set(GHOST_VERSION_IDENTIFIER alpha)
set(GHOST_VERSION "${GHOST_VERSION_MAJOR}.${GHOST_VERSION_MINOR}.${GHOST_VERSION_REVISION}")
if (DEFINED GHOST_VERSION_IDENTIFIER)
    set(GHOST_VERSION "${GHOST_VERSION}-${GHOST_VERSION_IDENTIFIER}")
endif()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

include(CheckCCompilerFlag)
include(${PROJECT_SOURCE_DIR}/cmake/OptimizeForArchitecture.cmake)

option(USE_MPI "Build with MPI" 1)
option(USE_OPENMP "Build with OpenMP" 1)
option(USE_CUDA "Build with CUDA" 1)
option(USE_SCOTCH "Use PT-Scotch for matrix re-ordering" 1)
option(USE_CUDA_PINNEDMEM "Use CUDA pinned memory" 0)
option(USE_LONGIDX "Enable 64-bit indices for matrices and vectors" 0)
option(BUILD_MIC "Build for Intel MIC" 0)
set(BLAS "MKL" CACHE STRING "BLAS library to be used (required). Valid values: MKL, GSL, LIBSCI")
set(LIBTYPE "SHARED" CACHE STRING "Type of GHOST build (required). Valid values: SHARED, STATIC")
set(VERBOSITY 0 CACHE STRING "Level of verbosity information. Valid values: 0 (print only warnings and errors), 1, 2, 3 (increasing level of verbosity)")
set(INSTRUMENTATION "" CACHE STRING "Type of instrumentation. Valid values: TIMING, LIKWID")

find_package(PkgConfig)

add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -DDEBUG=${VERBOSITY})

set (SOURCES
        src/bincrs.c
        src/bincrs.cpp
		src/context.c
        src/core.c
		src/crs.c 
		src/crs.cpp 
        src/densemat.c
		src/densemat_cm.c 
		src/densemat_rm.c 
		src/densemat_cm.cpp 
		src/densemat_rm.cpp 
        src/error.c
        src/instr.c
        src/locality.c
        src/machine.c
		src/math.c
        src/omp.c
        src/pumap.c
        src/rand.c
		src/sell.c 
		src/sell.cpp
		src/sparsemat.c
		src/sparsemat.cpp
        src/spmv_overlap.c 
		src/spmv_nompi.c 
        src/spmv_task.c
        src/spmv_vector.c 
		src/task.c
		src/taskq.c
        src/thpool.c
        src/timing.c
        src/timing.cpp
        src/types.c
		src/util.c 
	)

set (HEADERS
        include/ghost/bincrs.h
        include/ghost/bindensemat.h
		include/ghost/blas_mangle.h
		include/ghost/complex.h
		include/ghost/context.h
        include/ghost/core.h
		include/ghost/crs.h
		include/ghost/densemat.h
		include/ghost/densemat_cm.h
		include/ghost/densemat_rm.h
		include/ghost/error.h
        include/ghost/instr.h
		include/ghost/locality.h	
        include/ghost/log.h
		include/ghost/machine.h
		include/ghost/math.h
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
		include/ghost/types.h
		include/ghost/util.h
	)

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/src_pp/sell_kernel_intr.c 
        DEPENDS ${CMAKE_SOURCE_DIR}/src/sell_kernel_intr.c ${CMAKE_SOURCE_DIR}/bin/ghost_pp 
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/src_pp && cat ${CMAKE_SOURCE_DIR}/src/sell_kernel_intr.c | ${CMAKE_SOURCE_DIR}/bin/ghost_pp > ${CMAKE_BINARY_DIR}/src_pp/sell_kernel_intr.c
        )

add_custom_target(preprocess DEPENDS ${CMAKE_BINARY_DIR}/src_pp/sell_kernel_intr.c)


if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray") # Cray compiler cannot compile intrinsics
    set (SOURCES ${SOURCES} ${CMAKE_BINARY_DIR}/src_pp/sell_kernel_intr.c)
else()
    set(INTRINSICS_COMPILER "" CACHE PATH "Compiler to use for files containing compiler intrinsics (tested with icc)")
    add_custom_command(OUTPUT CMakeFiles/ghost.dir/src/sell_kernel_intr.c.o COMMAND ${INTRINSICS_COMPILER}  -DDEBUG=0 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -Dghost_EXPORTS -openmp -fPIC -O3 -I${CMAKE_SOURCE_DIR}/include/ -I${CMAKE_SOURCE_DIR}/include/ghost/ -I${CMAKE_BINARY_DIR}/ -I${CMAKE_BINARY_DIR}/ghost/ -o CMakeFiles/ghost.dir/src/sell_kernel_intr.c.o  -c ${CMAKE_SOURCE_DIR}/src/sell_kernel_intr.c) 
    set (SOURCES ${SOURCES} CMakeFiles/ghost.dir/src/sell_kernel_intr.c.o)
endif()

if (USE_CUDA)
	find_package(CUDA)
    if (CUDA_FOUND)
        set(GHOST_HAVE_CUDA 1)
        cuda_include_directories(${CMAKE_SOURCE_DIR}/include
                ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/ghost ${HWLOC_INCLUDE_DIR})
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC;-gencode=arch=compute_35,code=sm_35)
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND CUDA_NVCC_FLAGS -G -g)
        endif()
        cuda_compile(SELL_KERNEL_CU_O src/sell_kernel.cu)
        cuda_compile(VEC_CU_O src/densemat.cu)
        cuda_compile(CRS_CU_O src/crs.cu)
        set (SOURCES 
                ${SOURCES} 
                ${SELL_KERNEL_CU_O}
                ${VEC_CU_O}
                ${CRS_CU_O}
                src/cu_util.c
            )
        include_directories(${CUDA_INCLUDE_DIRS})
        set (HEADERS ${HEADERS}
                include/ghost/cu_util.h
                include/ghost/cu_densemat.h
            )
        if (USE_CUDA_PINNEDMEM)
            set(GHOST_HAVE_CUDA_PINNEDMEM 1)
        endif()
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
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
    set_property(TARGET ghost APPEND PROPERTY LINK_INTERFACE_LIBRARIES
            ${CUDA_LIBRARIES})
endif()

if (USE_SCOTCH)
    set(GHOST_HAVE_SCOTCH 1)
    set(SCOTCH_INCLUDE_DIR "${SCOTCH_INCLUDE_DIR}" CACHE PATH "Location of PT-Scotch headers")
    if (NOT EXISTS "${SCOTCH_INCLUDE_DIR}/scotch.h")
        message(FATAL_ERROR "Could not find scotch.h in SCOTCH_INCLUDE_DIR!")
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
    set_property(TARGET ghost APPEND PROPERTY LINK_INTERFACE_LIBRARIES ${SCOTCH_LIBRARIES})
    mark_as_advanced(LIBPTSCOTCH LIBSCOTCH LIBPTSCOTCHERR LIBSCOTCHERR)
endif()



if (USE_LONGIDX)
	set(GHOST_HAVE_LONGIDX 1)
    if (${BLAS} STREQUAL "MKL")
		add_definitions(-DMKL_ILP64)
	endif()
endif()

if (NOT ${INSTRUMENTATION} STREQUAL "NONE")
	if (${INSTRUMENTATION} STREQUAL "TIMING")
		set(GHOST_HAVE_INSTR_TIMING 1)
	endif()
	if (${INSTRUMENTATION} STREQUAL "LIKWID")
		set(GHOST_HAVE_INSTR_LIKWID 1)
	    set(LIKWID_INCLUDE_DIR "" CACHE PATH "Location of likwid.h")
		include_directories(${LIKWID_INCLUDE_DIR})
        add_definitions(-DLIKWID_PERFMON)
        get_filename_component(LIKWID_LIBDIR ${LIKWID_INCLUDE_DIR}/../lib REALPATH)
	    find_library(LIKWID_LIBRARIES likwid ${LIKWID_LIBDIR})
        set_property(TARGET ghost APPEND PROPERTY LINK_INTERFACE_LIBRARIES ${LIKWID_LIBRARIES})
	endif()
endif()

	

if (USE_OPENMP)	
	find_package(OpenMP REQUIRED)
	set(GHOST_HAVE_OPENMP 1)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#if (${BLAS} STREQUAL "MKL")
#    set(MKL_LIBDIR "" CACHE PATH "Location of libmkl")
#	link_directories(${MKL_LIBDIR})
#	link_libraries("-lmkl_core")
#	if (CMAKE_COMPILER_IS_GNUCC)
#		link_libraries("-lmkl_gf_lp64")
#	else ()
#		link_libraries("-mkl -lmkl_intel_lp64")
#	endif()
#elseif(${BLAS} STREQUAL "GSL")
#	find_package(GSL REQUIRED)
#	link_libraries(${GSL_LIBRARIES})
#elseif(${BLAS} STREQUAL "LIBSCI")
#	set(LIBSCI_LIBDIR "${LIBSCI_LIBDIR}" CACHE PATH "Location of LibSCI library")
#	set(LIBSCI_INCDIR "${LIBSCI_INCDIR}" CACHE PATH "Location of cblas.h")
#    include_directories(${LIBSCI_INCDIR})
#    link_directories(${LIBSCI_LIBDIR})
#    link_libraries("-lsci_cray_mp")
#endif()

if (${BLAS} STREQUAL "MKL")
	set(GHOST_HAVE_MKL 1)
    set(MKL_INCLUDE_DIR "" CACHE PATH "Location of mkl.h")
    if (NOT EXISTS "${MKL_INCLUDE_DIR}/mkl.h")
        message(FATAL_ERROR "Could not find mkl.h in MKL_INCLUDE_DIR!")
    endif()
	include_directories(SYSTEM ${MKL_INCLUDE_DIR})
    if (BUILD_MIC)
        get_filename_component(MKL_LIBDIR ${MKL_INCLUDE_DIR}/../lib/mic REALPATH)
    else ()
        get_filename_component(MKL_LIBDIR ${MKL_INCLUDE_DIR}/../lib/intel64 REALPATH)
    endif()

    set(LIBMKL_CORE LIBMKL_CORE-NOTFOUND)
    set(LIBMKL_X_THREAD LIBMKL_X_THREAD-NOTFOUND)
    set(LIBMKL_INTEL_XLP64 LIBMKL_INTEL_XLP64-NOTFOUND)
	
    find_library(LIBMKL_CORE mkl_core ${MKL_LIBDIR})
    
    if (USE_LONGIDX)
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

    set (BLAS_LIBRARIES ${LIBMKL_CORE} ${LIBMKL_INTEL_XLP64} ${LIBMKL_X_THREAD})
    mark_as_advanced(LIBMKL_CORE LIBMKL_INTEL_XLP64 LIBMKL_X_THREAD) 
elseif(${BLAS} STREQUAL "GSL")
	set(GHOST_HAVE_GSL 1)
	find_package(GSL REQUIRED)
	include_directories(${GSL_INCLUDE_DIRS}/gsl)
    set (BLAS_LIBRARIES ${GSL_LIBRARIES})
elseif(${BLAS} STREQUAL "LIBSCI")
	message(FATAL_ERROR "LibSci cmake not up-to-date")
    set(GHOST_HAVE_LIBSCI 1)
    find_package(PkgConfig)
    if ( PKGCONFIG_FOUND )
        find_package(LibSci REQUIRED)
    else()
        set(LIBSCI_INCLUDE_DIR "" CACHE PATH "Location of cblas.h")
    endif()
	include_directories(${LIBSCI_INCLUDE_DIR})
endif()

set_property(TARGET ghost APPEND PROPERTY LINK_INTERFACE_LIBRARIES ${BLAS_LIBRARIES})


if (PKGCONFIG_FOUND)
	find_package(HWLOC REQUIRED QUIET)
else()
	set(HWLOC_INCLUDE_DIR "" CACHE PATH "Location of hwloc.h")
endif()
   
set(LIBHWLOC LIBHWLOC-NOTFOUND) 
get_filename_component(HWLOC_LIBDIR ${HWLOC_INCLUDE_DIR}/../lib REALPATH)
find_library(LIBHWLOC hwloc ${HWLOC_LIBDIR})
set(HWLOC_LIBRARIES ${LIBHWLOC})
set_property(TARGET ghost APPEND PROPERTY LINK_INTERFACE_LIBRARIES ${HWLOC_LIBRARIES})

mark_as_advanced(LIBHWLOC)
mark_as_advanced(CLEAR HWLOC_INCLUDE_DIR)

if (NOT EXISTS "${HWLOC_INCLUDE_DIR}/hwloc.h")
	message(FATAL_ERROR "Could not find hwloc.h neither via pkg-config nor in the defined HWLOC_INCLUDEDIR!")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR} ${PROJECT_BINARY_DIR}/ghost ${HWLOC_INCLUDE_DIR})
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
# msg 193 complains about commented out lines in config.h
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -h c99 -h msglevel_3 -h nomessage=193")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -h msglevel_3 -h nomessage=193")
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -std=c99 -Wall -Wextra -pedantic -save-temps")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++0x -Wall -Wextra -pedantic -save-temps")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-variable -fsource-asm")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable -fsource-asm")
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-alias")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-alias")
         endif()
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") 
            if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -fstrict-aliasing")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fstrict-aliasing")
            elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
                set(CLANG_SANITIZE "" CACHE STRING "What should be sanitized by Clang for a Debug build? (Adress, Thread, Memory or leave empty)")
                if ("${CLANG_SANITIZE}" STREQUAL "Address")
                    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
                elseif ("${CLANG_SANITIZE}" STREQUAL "Thread")
                    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
                elseif ("${CLANG_SANITIZE}" STREQUAL "Memory")
                    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
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


if (USE_MPI)	
    set(MPI_LIBRARIES "-lmpi_mt")
	find_package(MPI)
	include_directories(${MPI_INCLUDE_PATH})
    foreach(item ${MPI_LIBRARIES})
        string(REGEX MATCH ".*libmpi.so" match ${item})
        if (match)
            get_filename_component(MPI_LIBDIR ${match} PATH)
            if (EXISTS "${MPI_LIBDIR}/libmpi_mt.so")
                list(REMOVE_ITEM MPI_LIBRARIES ${item})
                list(APPEND MPI_LIBRARIES ${MPI_LIBDIR}/libmpi_mt.so)
            endif()
        endif()
    endforeach()
#	link_libraries(${MPI_LIBRARIES})
	get_filename_component(MPI_LIBDIR "${MPI_LIBRARY}" PATH)
	find_library(LIBMPI_CXX mpi_cxx ${MPI_LIBDIR})
	if (LIBMPI_CXX)
        set_property(TARGET ghost APPEND PROPERTY LINK_INTERFACE_LIBRARIES ${LIBMPI_CXX})
#		link_libraries(${LIBMPI_CXX})
	endif()
    set_property(TARGET ghost APPEND PROPERTY LINK_INTERFACE_LIBRARIES ${MPI_LIBRARIES})
	set(GHOST_HAVE_MPI 1)
#set_target_properties(ghost PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#	set_target_properties(ghost PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
     mark_as_advanced(MPI_EXTRA_LIBRARY MPI_LIBRARY LIBMPI_CXX)
     mark_as_advanced(CLEAR MPI_C_INCLUDE_PATH)
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
configure_file(${CMAKE_SOURCE_DIR}/ghost.pc.in ${CMAKE_BINARY_DIR}/ghost.pc @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/ghost-config.cmake.in ${CMAKE_BINARY_DIR}/ghost-config.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/ghost-config-version.cmake.in ${CMAKE_BINARY_DIR}/ghost-config-version.cmake @ONLY)

if (${LIBTYPE} STREQUAL "STATIC")
    install(TARGETS ghost EXPORT ghost-targets ARCHIVE DESTINATION lib/ghost)
#    install(EXPORT ghost-targets DESTINATION lib/ghost)
elseif (${LIBTYPE} STREQUAL "SHARED")
    install(TARGETS ghost EXPORT ghost-targets LIBRARY DESTINATION lib/ghost)
#    install(EXPORT ghost-targets DESTINATION lib/ghost)
endif()
install(FILES ${CMAKE_BINARY_DIR}/ghost-config.cmake DESTINATION lib/ghost)
install(FILES ${CMAKE_BINARY_DIR}/ghost-config-version.cmake DESTINATION lib/ghost)
install(FILES include/ghost.h DESTINATION include)
install(FILES ${HEADERS} DESTINATION include/ghost)
install(FILES ${CMAKE_BINARY_DIR}/ghost.pc DESTINATION lib/pkgconfig)
install(FILES ${CMAKE_BINARY_DIR}/ghost/config.h DESTINATION include/ghost)

configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
	IMMEDIATE @ONLY)
add_custom_target(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake") 


find_package(Doxygen)
if (DOXYGEN_FOUND)
	configure_file(${CMAKE_SOURCE_DIR}/doxygen/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif()

