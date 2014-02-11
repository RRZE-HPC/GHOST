cmake_minimum_required (VERSION 2.8)
project (GHOST)

set(GHOST_VERSION_MAJOR 0)
set(GHOST_VERSION_MINOR 5)
set(GHOST_VERSION_REVISION 1)
set(GHOST_VERSION "${GHOST_VERSION_MAJOR}.${GHOST_VERSION_MINOR}.${GHOST_VERSION_REVISION}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

include(${PROJECT_SOURCE_DIR}/cmake/OptimizeForArchitecture.cmake)
include(CheckFunctionExists)

option(USE_MPI "Build with MPI" 1)
option(USE_OPENMP "Build with OpenMP" 1)
option(USE_CUDA "Build with CUDA" 1)
option(USE_CUDA_PINNEDMEM "Use CUDA pinned memory" 0)
option(USE_LONGIDX "Enable 64-bit indices for matrices and vectors" 0)
option(BUILD_MIC "Build for Intel MIC" 0)
set(BLAS "MKL" CACHE STRING "BLAS library to be used (may be MKL, GSL or LIBSCI)")
set(LIBTYPE "SHARED" CACHE STRING "Type of GHOST build (may be SHARED or STATIC)")
set(VERBOSITY 0 CACHE STRING "Level of printed DEBUG information")
set(INSTRUMENTATION "NONE" CACHE STRING "Type of instrumentation (maybe NONE, TIMING, LIKWID)")

check_function_exists(bswap_64 GHOST_HAVE_BSWAP)

add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -DDEBUG=${VERBOSITY})

set (SOURCES
		src/context.c
        src/core.c
		src/crs.c 
		src/crs.cpp 
        src/error.c
        src/io.c
        src/io.cpp
        src/locality.c
        src/machine.c
		src/mat.c
		src/math.c
        src/pumap.c
		src/sell.c 
		src/sell.cpp
        src/spmvm_goodfaithhybrid.c 
		src/spmvm_nompi.c 
        src/spmvm_taskmode.c
        src/spmvm_vectormode.c 
		src/task.c
		src/taskq.c
        src/thpool.c
        src/timing.c
        src/types.c
		src/util.c 
		src/vec.c 
		src/vec.cpp 
	)

set (HEADERS
		include/ghost/blas_mangle.h
		include/ghost/complex.h
		include/ghost/constants.h
		include/ghost/context.h
        include/ghost/core.h
		include/ghost/crs.h
		include/ghost/error.h
        include/ghost/io.h
		include/ghost/locality.h	
        include/ghost/log.h
		include/ghost/machine.h
		include/ghost/mat.h
		include/ghost/math.h
        include/ghost/pumap.h
		include/ghost/sell.h
		include/ghost/task.h
		include/ghost/taskq.h
        include/ghost/thpool.h
        include/ghost/timing.h
		include/ghost/types.h
		include/ghost/util.h
		include/ghost/vec.h
	)

if (USE_MPI)
	set (SOURCES ${SOURCES} 
		)
endif()

if (USE_CUDA)
	find_package(CUDA QUIET)
	if (CUDA_FOUND)
		set(GHOST_HAVE_CUDA 1)
		cuda_include_directories(${CMAKE_SOURCE_DIR}/include
				${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/ghost)
		list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC;-gencode=arch=compute_35,code=sm_35)
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		    list(APPEND CUDA_NVCC_FLAGS -G -g)
        endif()
		cuda_compile(SELL_KERNEL_CU_O src/sell_kernel.cu)
		cuda_compile(VEC_CU_O src/vec.cu)
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
                include/ghost/cu_vec.h
                include/ghost/cu_crs.h
			)
        if (USE_CUDA_PINNEDMEM)
            set(GHOST_HAVE_CUDA_PINNEDMEM 1)
        endif()
	endif()
endif()

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray") # Cray compiler cannot compile intrinsics
    set (SOURCES ${SOURCES} src/sell_kernel_intr.c)
else()
    set(INTRINSICS_COMPILER "" CACHE PATH "Compiler to use for files containing compiler intrinsics (tested with icc)")
    add_custom_command(OUTPUT CMakeFiles/ghost.dir/src/sell_kernel_intr.c.o COMMAND ${INTRINSICS_COMPILER}  -DDEBUG=0 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -Dghost_EXPORTS -openmp -fPIC -O3 -I${CMAKE_SOURCE_DIR}/include/ -I${CMAKE_SOURCE_DIR}/include/ghost/ -I${CMAKE_BINARY_DIR}/ -I${CMAKE_BINARY_DIR}/ghost/ -o CMakeFiles/ghost.dir/src/sell_kernel_intr.c.o  -c ${CMAKE_SOURCE_DIR}/src/sell_kernel_intr.c) 
    set (SOURCES ${SOURCES} CMakeFiles/ghost.dir/src/sell_kernel_intr.c.o)
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
    set(MKL_INCLUDE_DIR "" CACHE PATH "Location of mkl.h")
	include_directories(SYSTEM ${MKL_INCLUDE_DIR})
elseif(${BLAS} STREQUAL "GSL")
	set(GHOST_HAVE_GSL 1)
	find_package(GSL REQUIRED)
	include_directories(${GSL_INCLUDE_DIRS}/gsl)
elseif(${BLAS} STREQUAL "LIBSCI")
	set(GHOST_HAVE_LIBSCI 1)
    find_package(PkgConfig)
    if ( PKGCONFIG_FOUND )
        find_package(LibSci REQUIRED)
    else()
        set(LIBSCI_INCLUDE_DIR "" CACHE PATH "Location of cblas.h")
    endif()
	include_directories(${LIBSCI_INCLUDE_DIR})
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

include_directories(${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR} ${PROJECT_BINARY_DIR}/ghost ${HWLOC_INCLUDE_DIR})
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
# msg 193 complains about commented out lines in config.h
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -h c99 -h msglevel_3 -h nomessage=193")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -h msglevel_3 -h nomessage=193")
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -std=c99 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++0x -Wall -Wextra -pedantic")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-variable")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable")
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
	find_package(MPI REQUIRED)
	set(GHOST_HAVE_MPI 1)
	include_directories(include/mpi ${MPI_INCLUDE_PATH})
	set_target_properties(ghost PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
	set_target_properties(ghost PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()


if (BUILD_MIC)
	set(GHOST_HAVE_MIC 1)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmic")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
elseif (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
	AutodetectHostArchitecture()
	OptimizeForArchitecture()
    if (AVX_FOUND)
	    set(GHOST_HAVE_AVX 1)
    endif()
    if (SSE_FOUND)
	    set(GHOST_HAVE_SSE 1)
    endif()

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
endif()

configure_file (
	"${PROJECT_SOURCE_DIR}/include/ghost/config.h.in"
	"${PROJECT_BINARY_DIR}/ghost/config.h"
)

if (${LIBTYPE} STREQUAL "STATIC")
install(TARGETS ghost EXPORT GHOST-targets ARCHIVE DESTINATION lib/ghost)
elseif (${LIBTYPE} STREQUAL "SHARED")
install(TARGETS ghost EXPORT GHOST-targets LIBRARY DESTINATION lib/ghost)
endif()
install(FILES GHOST-config.cmake DESTINATION lib/ghost)
install(FILES include/ghost.h DESTINATION include)
install(FILES ${HEADERS} DESTINATION include/ghost)
install(FILES ${CMAKE_BINARY_DIR}/ghost.pc DESTINATION lib/pkgconfig)
install(FILES ${CMAKE_BINARY_DIR}/ghost/config.h DESTINATION include/ghost)

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

