#ifndef GHOST_UTIL_H
#define GHOST_UTIL_H

#include "config.h"
#include "types.h"

#if GHOST_HAVE_CUDA
#include "cu_util.h"
#endif

#include <stdio.h>

#ifndef __cplusplus
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#else
#include <complex>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#endif
/******************************************************************************/
/****** Makros ****************************************************************/
/******************************************************************************/
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"



#ifdef GHOST_HAVE_MPI
#define ABORT(...) {\
    LOG(ABORT,ANSI_COLOR_MAGENTA,__VA_ARGS__)\
    MPI_safecall(MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE));\
    exit(EXIT_FAILURE);\
}
#else
#define ABORT(...) {\
    LOG(ABORT,ANSI_COLOR_MAGENTA,__VA_ARGS__)\
    exit(EXIT_FAILURE);\
}
#endif



#define MPI_safecall(call) {\
    int mpierr = call ;\
    if( MPI_SUCCESS != mpierr ){\
        fprintf(stderr, ANSI_COLOR_RED "MPI error at %s:%d, %d\n" ANSI_COLOR_RESET,\
                __FILE__, __LINE__, mpierr);\
        fflush(stderr);\
    }\
}

#ifdef GHOST_HAVE_MPI
#define CU_safecall(call) {\
    cudaError_t __cuerr = call ;\
    if( cudaSuccess != __cuerr ){\
        int __me;\
        MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
        fprintf(stdout, ANSI_COLOR_RED "PE%d: CUDA error at %s:%d, %s\n" ANSI_COLOR_RESET,\
                __me, __FILE__, __LINE__, cudaGetErrorString(__cuerr));\
        fflush(stdout);\
    }\
}
#define CUBLAS_safecall(call) {\
    cublasStatus_t __stat = call ;\
    if( CUBLAS_STATUS_SUCCESS != __stat ){\
        int __me;\
        MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
        fprintf(stdout, ANSI_COLOR_RED "PE%d: CUBLAS error at %s:%d: %d\n" ANSI_COLOR_RESET,\
                __me, __FILE__, __LINE__, __stat);\
        fflush(stdout);\
    }\
}
#define CURAND_safecall(call) {\
    curandStatus_t __stat = call ;\
    if( CURAND_STATUS_SUCCESS != __stat ){\
        int __me;\
        MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
        fprintf(stdout, ANSI_COLOR_RED "PE%d: CURAND error at %s:%d: %d\n" ANSI_COLOR_RESET,\
                __me, __FILE__, __LINE__, __stat);\
        fflush(stdout);\
    }\
}

#define CU_checkerror() {\
    cudaError_t __cuerr = cudaGetLastError();\
    if( cudaSuccess != __cuerr ){\
        int __me;\
        MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
        fprintf(stdout, ANSI_COLOR_RED "PE%d: CUDA error at %s:%d, %s\n" ANSI_COLOR_RESET,\
                __me, __FILE__, __LINE__, cudaGetErrorString(__cuerr));\
        fflush(stdout);\
    }\
}

#else

#define CU_safecall(call) {\
    cudaError_t __cuerr = call ;\
    if( cudaSuccess != __cuerr ){\
        fprintf(stdout, ANSI_COLOR_RED "CUDA error at %s:%d, %s\n" ANSI_COLOR_RESET,\
                __FILE__, __LINE__, cudaGetErrorString(__cuerr));\
        fflush(stdout);\
    }\
}
#define CUBLAS_safecall(call) {\
    cublasStatus_t __stat = call ;\
    if( CUBLAS_STATUS_SUCCESS != __stat ){\
        fprintf(stdout, ANSI_COLOR_RED "CUBLAS error at %s:%d: %d\n" ANSI_COLOR_RESET,\
                __FILE__, __LINE__, __stat);\
        fflush(stdout);\
    }\
}
#define CURAND_safecall(call) {\
    curandStatus_t __stat = call ;\
    if( CURAND_STATUS_SUCCESS != __stat ){\
        fprintf(stdout, ANSI_COLOR_RED "CURAND error at %s:%d: %d\n" ANSI_COLOR_RESET,\
                __FILE__, __LINE__, __stat);\
        fflush(stdout);\
    }\
}

#define CU_checkerror() {\
    cudaError_t __cuerr = cudaGetLastError();\
    if( cudaSuccess != __cuerr ){\
        fprintf(stdout, ANSI_COLOR_RED "CUDA error at %s:%d, %s\n" ANSI_COLOR_RESET,\
                __FILE__, __LINE__, cudaGetErrorString(__cuerr));\
        fflush(stdout);\
    }\
}
#endif

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)<(y)?(y):(x))
#endif


#define GHOST_VTRAITS_INIT(...) {.flags = GHOST_VEC_DEFAULT, .aux = NULL, .datatype = GHOST_DT_DOUBLE|GHOST_DT_REAL, .nrows = 0, .nrowshalo = 0, .nrowspadded = 0, .nvecs = 1, .localdot = NULL, ## __VA_ARGS__ }

#define GHOST_MTRAITS_INIT(...) {.flags = GHOST_SPM_DEFAULT, .aux = NULL, .nAux = 0, .datatype = GHOST_DT_DOUBLE|GHOST_DT_REAL, .format = GHOST_SPM_FORMAT_CRS, .shift = NULL, .scale = NULL, ## __VA_ARGS__ }


#define GHOST_TIME(_niter,_func,...)\
    double _func ## _start, _func ## _end, _func ## _tstart;\
    double _func ## _tmin = DBL_MAX;\
    double _func ## _tmax = 0.;\
    double _func ## _tavg = 0.;\
    int _func ## _it;\
    _func ## _tstart=ghost_wctime();\
    for (_func ## _it=0; _func ## _it<_niter; _func ## _it++) {\
       _func ## _start = ghost_wctime();\
       _func(__VA_ARGS__);\
       _func ## _end = ghost_wctime();\
       _func ## _tmin = MIN(_func ## _end-_func ## _start,_func ## _tmin);\
       _func ## _tmin = MAX(_func ## _end-_func ## _start,_func ## _tmin);\
    }\
    _func ## _tavg = (ghost_wctime()-_func ## _tstart)/((double)_niter);




#define UNUSED(x) (void)(x)
/******************************************************************************/


#ifdef __cplusplus
extern "C" {
#endif

#ifdef GHOST_HAVE_MPI
#else
#endif
    extern int hasCUDAdevice;
//    extern ghost_type_t ghost_type; 

    void ghost_printHeader(const char *fmt, ...);
    void ghost_printFooter(); 
    void ghost_printLine(const char *label, const char *unit, const char *format, ...);
    ghost_error_t ghost_printSysInfo();
    ghost_error_t ghost_printGhostInfo();



    char * ghost_workdistName(int ghostOptions);
    char * ghost_modeName(int spmvmOptions);
    char * ghost_datatypeName(int datatype);
    char * ghost_symmetryName(int symmetry);

    /**
     * @brief Pad a number to a multiple of a second number.
     *
     * @param nrows The number to be padded.
     * @param padding The desired padding.
     *
     * @return nrows padded to a multiple of padding or nrows if padding is 0.
     */
    ghost_midx_t ghost_pad(ghost_midx_t nrows, ghost_midx_t padding);

    size_t ghost_sizeofDataType(int dt);
    int ghost_datatypeValid(int datatype);
    int ghost_symmetryValid(int symmetry);
    int ghost_archIsBigEndian();
    char ghost_datatypePrefix(int dt);
    int ghost_dataTypeIdx(int datatype);
    int ghost_getSpmvmModeIdx(int spmvmOptions);

    double ghost_wctime();
    double ghost_wctimemilli();

    void *ghost_malloc(const size_t size);
    void *ghost_malloc_align(const size_t size, const size_t align);
    int ghost_flopsPerSpmvm(int m_t, int v_t);
    
    void ghost_ompSetNumThreads(int nthreads);
    int ghost_ompGetThreadNum();
    int ghost_ompGetNumThreads();
    ghost_error_t ghost_mpi_datatype(ghost_mpi_datatype_t *dt, int datatype);
    ghost_error_t ghost_mpi_op_sum(ghost_mpi_op_t * op, int datatype);

    /**
     * @brief Computes a hash from three integral input values.
     *
     * @param a First parameter.
     * @param b Second parameter.
     * @param c Third parameter.
     *
     * @return Hash value.
     *
     * The function has been taken from http://burtleburtle.net/bob/hash/doobs.html
     */
    int ghost_hash(int a, int b, int c);

#ifdef __cplusplus
}
#endif
#endif
