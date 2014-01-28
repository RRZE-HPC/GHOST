#ifndef __GHOST_UTIL_H__
#define __GHOST_UTIL_H__

#include "config.h"
#include "types.h"

#ifdef GHOST_HAVE_MPI
#include "mpi_util.h"
#endif

#ifdef GHOST_HAVE_OPENCL
#include "ghost_cl_util.h"
#endif

#ifdef GHOST_HAVE_CUDA
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

//#define DEBUG_IDT 0
//extern int DEBUG_IDT;

//#define DEBUG_INDENT DEBUG_IDT+=2
//#define DEBUG_OUTDENT DEBUG_IDT-=2

#define IF_DEBUG(level) if(DEBUG >= level)

#define FILE_BASENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

/* stolen from http://stackoverflow.com/a/11172679 */
/* expands to the first argument */
#define FIRST(...) FIRST_HELPER(__VA_ARGS__, throwaway)
#define FIRST_HELPER(first, ...) first

/*
 * if there's only one argument, expands to nothing.  if there is more
 * than one argument, expands to a comma followed by everything but
 * the first argument.  only supports up to 9 arguments but can be
 * trivially expanded.
 */
#define REST(...) REST_HELPER(NUM(__VA_ARGS__), __VA_ARGS__)
#define REST_HELPER(qty, ...) REST_HELPER2(qty, __VA_ARGS__)
#define REST_HELPER2(qty, ...) REST_HELPER_##qty(__VA_ARGS__)
#define REST_HELPER_ONE(first)
#define REST_HELPER_TWOORMORE(first, ...) , __VA_ARGS__
#define NUM(...) \
    SELECT_10TH(__VA_ARGS__, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE,\
            TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, ONE, throwaway)
#define SELECT_10TH(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, ...) a10

#ifdef GHOST_HAVE_MPI
#define LOG(type,color,...) {\
    int __me;\
    MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
    fprintf(stderr, color "PE%d " #type " at %s() <%s:%d>: " FIRST(__VA_ARGS__) ANSI_COLOR_RESET "\n", __me, __func__, FILE_BASENAME, __LINE__ REST(__VA_ARGS__)); \
    fflush(stderr);\
    }
#else
#define LOG(type,color,...) {\
    fprintf(stderr, color #type " at %s() <%s:%d>: " FIRST(__VA_ARGS__) ANSI_COLOR_RESET "\n", __func__, FILE_BASENAME, __LINE__ REST(__VA_ARGS__));\
    }
#endif


#define DEBUG_LOG(level,...) {if(DEBUG >= level) { LOG(DEBUG,ANSI_COLOR_RESET,__VA_ARGS__) }}
#define INFO_LOG(...) LOG(INFO,ANSI_COLOR_BLUE,__VA_ARGS__)
#define WARNING_LOG(...) LOG(WARNING,ANSI_COLOR_YELLOW,__VA_ARGS__)
#define ERROR_LOG(...) LOG(ERROR,ANSI_COLOR_RED,__VA_ARGS__)

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
#define CL_safecall(call) {\
    cl_int clerr = call ;\
    if( CL_SUCCESS != clerr ){\
        int __me;\
        MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
        fprintf(stderr, ANSI_COLOR_RED "PE%d: OpenCL error at %s:%d, %s\n" ANSI_COLOR_RESET,\
                __me, __FILE__, __LINE__, CL_errorString(clerr));\
        fflush(stderr);\
    }\
}

#define CL_checkerror(clerr) do{\
    if( CL_SUCCESS != clerr ){\
        int __me;\
        MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
        fprintf(stdout, ANSI_COLOR_RED "PE%d: OpenCL error at %s:%d, %s\n" ANSI_COLOR_RESET,\
                __me, __FILE__, __LINE__, CL_errorString(clerr));\
        fflush(stdout);\
    }\
} while(0)
#else
#define CL_safecall(call) {\
    cl_int clerr = call ;\
    if( CL_SUCCESS != clerr ){\
        fprintf(stderr, ANSI_COLOR_RED "OpenCL error at %s:%d, %s\n" ANSI_COLOR_RESET,\
                __FILE__, __LINE__, CL_errorString(clerr));\
        fflush(stderr);\
    }\
}

#define CL_checkerror(clerr) do{\
    if( CL_SUCCESS != clerr ){\
        fprintf(stdout, ANSI_COLOR_RED "OpenCL error at %s:%d, %s\n" ANSI_COLOR_RESET,\
                __FILE__, __LINE__, CL_errorString(clerr));\
        fflush(stdout);\
    }\
} while(0)
#endif

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

#define GHOST_REGISTER_DT_D(name) \
    typedef double name ## _t; \
int name = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL; \

#define GHOST_REGISTER_DT_S(name) \
    typedef float name ## _t; \
int name = GHOST_BINCRS_DT_FLOAT|GHOST_BINCRS_DT_REAL; \

#define GHOST_REGISTER_DT_C(name) \
    typedef complex float name ## _t; \
int name = GHOST_BINCRS_DT_FLOAT|GHOST_BINCRS_DT_COMPLEX; \

#define GHOST_REGISTER_DT_Z(name) \
    typedef complex double name ## _t; \
int name = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_COMPLEX; \

#define GHOST_VTRAITS_INIT(...) {.flags = GHOST_VEC_DEFAULT, .aux = NULL, .datatype = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL, .nrows = 0, .nrowshalo = 0, .nrowspadded = 0, .nvecs = 1, .localdot = NULL, ## __VA_ARGS__ }

#define GHOST_MTRAITS_INIT(...) {.flags = GHOST_SPM_DEFAULT, .aux = NULL, .nAux = 0, .datatype = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL, .format = GHOST_SPM_FORMAT_CRS, .shift = NULL, .scale = NULL, ## __VA_ARGS__ }

#if GHOST_HAVE_INSTR_TIMING

#define GHOST_INSTR_START(tag) double __start_##tag = ghost_wctime();
#define GHOST_INSTR_STOP(tag) printf(ANSI_COLOR_BLUE "[GHOST_TIMING] %s: %e secs\n" ANSI_COLOR_RESET,\
#tag,ghost_wctime()-__start_##tag);

#elif GHOST_HAVE_INSTR_LIKWID

#include <likwid.h>
#define GHOST_INSTR_START(tag) {\
_Pragma("omp parallel")\
    LIKWID_MARKER_START(#tag);}
#define GHOST_INSTR_STOP(tag) {\
_Pragma("omp parallel")\
    LIKWID_MARKER_STOP(#tag);}

#else

#define GHOST_INSTR_START(tag)
#define GHOST_INSTR_STOP(tag)

#endif

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
    extern MPI_Datatype GHOST_MPI_DT_C;
    extern MPI_Op GHOST_MPI_OP_SUM_C;
    extern MPI_Datatype GHOST_MPI_DT_Z;
    extern MPI_Op GHOST_MPI_OP_SUM_Z;
    extern MPI_Comm ghost_node_comm;
    extern int ghost_node_rank;
#else
    extern int ghost_node_comm;
    extern int ghost_node_rank;
#endif
    extern int hasCUDAdevice;
    extern int hasOPENCLdevice;
    extern ghost_type_t ghost_type; 

    void ghost_printHeader(const char *fmt, ...);
    void ghost_printFooter(); 
    void ghost_printLine(const char *label, const char *unit, const char *format, ...);
    void ghost_printContextInfo(ghost_context_t *context);
    void ghost_printMatrixInfo(ghost_mat_t *matrix);
    void ghost_printSysInfo();
    void ghost_printGhostInfo();


    void ghost_solver_nompi(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
    void ghost_referenceSolver(ghost_vec_t *, char *matrixPath, int datatype, ghost_vec_t *rhs, int nIter, int spmvmOptions);

    char * ghost_workdistName(int ghostOptions);
    char * ghost_modeName(int spmvmOptions);
    char * ghost_datatypeName(int datatype);
    char * ghost_symmetryName(int symmetry);

    int ghost_pad(int nrows, int padding);

//    void ghost_freeCommunicator( ghost_comm_t* const );
    size_t ghost_sizeofDataType(int dt);
    int ghost_datatypeValid(int datatype);
    int ghost_symmetryValid(int symmetry);
    int ghost_archIsBigEndian();
    void ghost_pickSpMVMMode(ghost_context_t * context, int *spmvmOptions);
    char ghost_datatypePrefix(int dt);
    int ghost_dataTypeIdx(int datatype);
    ghost_midx_t ghost_globalIndex(ghost_context_t *, ghost_midx_t);

    int ghost_getSpmvmModeIdx(int spmvmOptions);
    double ghost_wctime();
    double ghost_wctimemilli();

    // this function is used for thread-safe random number generation.
    // The state for each OpenMP thread is initialized in ghost_init/ghost_rand_init
    unsigned int* ghost_getRandState();

    double ghost_bench_spmvm(ghost_context_t *context, ghost_vec_t *res, ghost_mat_t *mat, ghost_vec_t *invec, int *spmvmOptions, int nIter);
    void *ghost_malloc(const size_t size);
    void *ghost_malloc_align(const size_t size, const size_t align);
    int ghost_flopsPerSpmvm(int m_t, int v_t);
    ghost_vtraits_t * ghost_cloneVtraits(ghost_vtraits_t *t1);
    void ghost_ompSetNumThreads(int nthreads);
    int ghost_ompGetThreadNum();
    int ghost_ompGetNumThreads();
    int ghost_init(int argc, char **argv);
    void ghost_finish();
    size_t ghost_getSizeOfLLC();

    int ghost_setType(ghost_type_t t);
    int ghost_setHybridMode(ghost_hybridmode_t hm);
    int ghost_hash(int a, int b, int c);

#ifdef __cplusplus
}
#endif
#endif
