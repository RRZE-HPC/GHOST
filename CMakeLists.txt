cmake_minimum_required (VERSION 2.8)
project (GHOST)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

include(${PROJECT_SOURCE_DIR}/cmake/OptimizeForArchitecture.cmake)

option(USE_MPI "Build with MPI" 1)
option(USE_OPENMP "Build with OpenMP" 1)
option(USE_CUDA "Build with CUDA" 0)
option(USE_OPENCL "Build with OpenCL" 0)
option(LONGIDX "Enable 64-bit indices for matrices and vectors" 0)
set(BLAS "MKL" CACHE STRING "BLAS library (currently only MKL supported)")
set(VERBOSITY 0 CACHE STRING "Level of printed DEBUG information")

add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -DDEBUG=${VERBOSITY})

set (SOURCES 
		src/ghost.c 
		src/ghost_util.c 
		src/ghost_mat.c 
		src/ghost_solver_nompi.c 
		src/ghost_vec.c 
		src/ghost_taskq.c 
		src/crs.c 
		src/crs_kernel_tmpl.cpp 
		src/sell.c 
		src/sell_kernel_intr.c 
		src/ghost_vec_tmpl.cpp 
	src/sell_kernel_tmpl.cpp
		)

set (HEADERS
		include/crs.h
		include/ghost.h
		include/ghost_blas_mangle.h
		include/ghost_complex.h
		include/ghost_config.h
		include/ghost_constants.h
		include/ghost_mat.h
		include/ghost_taskq.h
		include/ghost_types.h
		include/ghost_util.h
		include/ghost_vec.h
		include/sell.h
	)

if (USE_MPI)
	set (SOURCES ${SOURCES} 
			src/ghost_mpi_util.c 
			src/ghost_solver_vectormode.c 
			src/ghost_solver_goodfaithhybrid.c 
			src/ghost_solver_taskmode.c
		)
	set (HEADERS ${HEADERS}
			include/ghost_mpi_util.h
		)
endif()

if (USE_CUDA)
	find_package(CUDA REQUIRED)
	set(GHOST_HAVE_CUDA 1)
	cuda_include_directories(${CMAKE_SOURCE_DIR}/include)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)
	cuda_compile(BJDS_KERNEL_O src/bjds_cukernel.cu)
	set (SOURCES ${SOURCES} 
			${BJDS_KERNEL_O}
			src/bjds_cukernel.cu	
		)
	include_directories(${CUDA_INCLUDE_DIRS})
	set (HEADERS ${HEADERS}
			include/ghost_cu_util.h
		)
endif()

add_library(ghost SHARED ${SOURCES})

if (LONGIDX)
	add_definitions(-DLONGIDX -DMKL_ILP64)
endif()

if (USE_OPENMP)	
	find_package(OpenMP REQUIRED)
	set(GHOST_HAVE_OPENMP 1)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (BLAS STREQUAL "MKL")
	set(GHOST_HAVE_MKL 1)
	include_directories($ENV{MKLROOT}/include)
endif()

find_package(HWLOC REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include ${HWLOC_INCLUDEDIR})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

if (USE_MPI)	
	find_package(MPI REQUIRED)
	set(GHOST_HAVE_MPI 1)
	target_link_libraries(ghost ${MPI_LIBRARIES})
	include_directories(include/mpi ${MPI_INCLUDE_PATH})
	set_target_properties(ghost PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
	set_target_properties(ghost PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

configure_file (
	"${PROJECT_SOURCE_DIR}/include/ghost_config.h.in"
	"${PROJECT_SOURCE_DIR}/include/ghost_config.h"
)

AutodetectHostArchitecture()
OptimizeForArchitecture()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")

install(TARGETS ghost LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/ghost)

configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
	IMMEDIATE @ONLY)
add_custom_target(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake") 
