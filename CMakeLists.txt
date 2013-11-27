cmake_minimum_required (VERSION 2.8)
project (GHOST)

set(GHOST_VERSION_MAJOR 0)
set(GHOST_VERSION_MINOR 5)
set(GHOST_VERSION_BUILD 0)
set(PACKAGE_VERSION "${GHOST_VERSION_MAJOR}.${GHOST_VERSION_MINOR}.${GHOST_VERSION_BUILD}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

include(${PROJECT_SOURCE_DIR}/cmake/OptimizeForArchitecture.cmake)

option(USE_MPI "Build with MPI" 1)
option(USE_OPENMP "Build with OpenMP" 1)
option(USE_CUDA "Build with CUDA" 1)
option(USE_OPENCL "Build with OpenCL" 1)
option(LONGIDX "Enable 64-bit indices for matrices and vectors" 0)
option(BUILD_MIC "Build for Intel MIC" 0)
set(BLAS "MKL" CACHE STRING "BLAS library to be used (may be MKL or GSL)")
set(VERBOSITY 0 CACHE STRING "Level of printed DEBUG information")
set(INSTRUMENTATION "NONE" CACHE STRING "Type of instrumentation (maybe NONE, TIMING, LIKWID)")

add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -DDEBUG=${VERBOSITY})

set (SOURCES 
		src/ghost.c 
		src/util.c 
		src/spmvm_nompi.c 
		src/vec.c 
		src/tasking.c 
		src/crs.c 
		src/crs.cpp 
		src/sell.c 
		src/sell_kernel_intr.c 
		src/vec.cpp 
		src/sell.cpp
		src/context.c
		src/mat.c
		src/math.c
		src/affinity.c
	)

set (HEADERS
		include/ghost.h
		include/ghost_affinity.h	
		include/ghost_blas_mangle.h
		include/ghost_complex.h
		${PROJECT_BINARY_DIR}/ghost_config.h
		include/ghost_constants.h
		include/ghost_context.h
		include/ghost_crs.h
		include/ghost_mat.h
		include/ghost_math.h
		include/ghost_sell.h
		include/ghost_taskq.h
		include/ghost_types.h
		include/ghost_util.h
		include/ghost_vec.h
	)

if (USE_MPI)
	set (SOURCES ${SOURCES} 
			src/mpi_util.c 
			src/spmvm_vectormode.c 
			src/spmvm_goodfaithhybrid.c 
			src/spmvm_taskmode.c
		)
	set (HEADERS ${HEADERS}
			include/ghost_mpi_util.h
		)
endif()

if (USE_CUDA)
	find_package(CUDA QUIET)
	if (CUDA_FOUND)
		set(GHOST_HAVE_CUDA 1)
		cuda_include_directories(${CMAKE_SOURCE_DIR}/include
				${CMAKE_BINARY_DIR})
		list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC;-gencode=arch=compute_35,code=sm_35)
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		    list(APPEND CUDA_NVCC_FLAGS -G -g)
        endif()
		cuda_compile(SELL_KERNEL_O src/sell_kernel.cu)
		set (SOURCES 
				${SOURCES} 
				${SELL_KERNEL_O}
				src/cu_util.c	
			)
		include_directories(${CUDA_INCLUDE_DIRS})
		set (HEADERS ${HEADERS}
				include/ghost_cu_util.h
			)
	endif()
endif()

if (USE_OPENCL)
	find_package(OpenCL)
	if (OPENCL_FOUND)
		set(GHOST_HAVE_OPENCL 1)
		set (SOURCES 
				${SOURCES} 
				src/ghost_cl_util.c	
			)
		include_directories(${OPENCL_INCLUDE_DIRS})
		set (HEADERS ${HEADERS}
				include/ghost_cl_util.h
			)
	endif()
endif()

add_library(ghost SHARED ${SOURCES})

if (LONGIDX)
	add_definitions(-DLONGIDX)
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
	endif()
endif()

	

if (USE_OPENMP)	
	find_package(OpenMP REQUIRED)
	set(GHOST_HAVE_OPENMP 1)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (${BLAS} STREQUAL "MKL")
	set(GHOST_HAVE_MKL 1)
	include_directories($ENV{MKLROOT}/include)
elseif(${BLAS} STREQUAL "GSL")
	set(GHOST_HAVE_GSL 1)
	find_package(GSL REQUIRED)
	include_directories(${GSL_INCLUDE_DIRS}/gsl)
endif()


find_package(PkgConfig)
if ( PKGCONFIG_FOUND )
	find_package(HWLOC REQUIRED)
else()
	set(HWLOC_INCLUDE_DIR "" CACHE PATH "Location of hwloc.h")
endif()

#if (NOT EXISTS "${HWLOC_INCLUDEDIR}/hwloc.h")
#	message(FATAL_ERROR "Could not find hw#loc.h neither via pkg-config nor in the defined HWLOC_INCLUDEDIR!")
#endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR} ${HWLOC_INCLUDE_DIR})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

if (USE_MPI)	
	find_package(MPI REQUIRED)
	set(GHOST_HAVE_MPI 1)
	include_directories(include/mpi ${MPI_INCLUDE_PATH})
	set_target_properties(ghost PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
	set_target_properties(ghost PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

configure_file (
	"${PROJECT_SOURCE_DIR}/include/ghost_config.h.in"
	"${PROJECT_BINARY_DIR}/ghost_config.h"
)

if (BUILD_MIC)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmic")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
elseif (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	AutodetectHostArchitecture()
	OptimizeForArchitecture()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
endif()


install(TARGETS ghost EXPORT GHOST-targets LIBRARY DESTINATION lib/ghost)
install(FILES GHOST-config.cmake DESTINATION lib/ghost)
install(FILES ${HEADERS} DESTINATION include/ghost)
install(FILES ${CMAKE_BINARY_DIR}/ghost.pc DESTINATION lib/pkgconfig)

configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
	IMMEDIATE @ONLY)
add_custom_target(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake") 

configure_file(${CMAKE_SOURCE_DIR}/ghost.pc.in ${CMAKE_BINARY_DIR}/ghost.pc @ONLY)

find_package(Doxygen)
if (DOXYGEN_FOUND)
	configure_file(${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif()

