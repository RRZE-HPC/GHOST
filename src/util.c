#define _GNU_SOURCE
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/vec.h"
#include "ghost/context.h"
#include "ghost/mat.h"
#include "ghost/math.h"
#include "ghost/task.h"
#include "ghost/constants.h"
#include "ghost/affinity.h"
#include "ghost/machine.h"
#include "ghost/io.h"
#include "ghost/log.h"
#include <libgen.h>
#include <unistd.h>
#include <byteswap.h>

#include <errno.h>
#if GHOST_HAVE_OPENMP
#include <omp.h>
#endif
#include <string.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <dirent.h>
#include <dlfcn.h>

//#define PRETTYPRINT

#define PRINTWIDTH 80
#define LABELWIDTH 40

#ifdef PRETTYPRINT
#define PRINTSEP "┊"
#else
#define PRINTSEP ":"
#endif

#define VALUEWIDTH (PRINTWIDTH-LABELWIDTH-(int)strlen(PRINTSEP))


extern char ** environ;

double ghost_wctime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec + tp.tv_usec/1000000.0);
}

double ghost_wctimemilli()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec * 1000. + tp.tv_usec/1000.0);
}

void ghost_printHeader(const char *fmt, ...)
{
    char label[1024] = "";

    if (fmt != NULL) {
        va_list args;
        va_start(args,fmt);
        vsnprintf(label,1024,fmt,args);
        va_end(args);
    }

    const int spacing = 4;
    int len = strlen(label);
    int nDash = (PRINTWIDTH-2*spacing-len)/2;
    int rem = (PRINTWIDTH-2*spacing-len)%2;
    int i;
#ifdef PRETTYPRINT
    printf("┌");
    for (i=0; i<PRINTWIDTH-2; i++) printf("─");
    printf("┐");
    printf("\n");
    printf("├");
    for (i=0; i<nDash-1; i++) printf("─");
    for (i=0; i<spacing; i++) printf(" ");
    printf("%s",label);
    for (i=0; i<spacing+rem; i++) printf(" ");
    for (i=0; i<nDash-1; i++) printf("─");
    printf("┤");
    printf("\n");
    printf("├");
    for (i=0; i<LABELWIDTH; i++) printf("─");
    printf("┬");
    for (i=0; i<VALUEWIDTH; i++) printf("─");
    printf("┤");
    printf("\n");
#else
    for (i=0; i<PRINTWIDTH; i++) printf("-");
    printf("\n");
    for (i=0; i<nDash; i++) printf("-");
    for (i=0; i<spacing; i++) printf(" ");
    printf("%s",label);
    for (i=0; i<spacing+rem; i++) printf(" ");
    for (i=0; i<nDash; i++) printf("-");
    printf("\n");
    for (i=0; i<PRINTWIDTH; i++) printf("-");
    printf("\n");
#endif
}

void ghost_printFooter() 
{
    int i;
#ifdef PRETTYPRINT
    printf("└");
    for (i=0; i<LABELWIDTH; i++) printf("─");
    printf("┴");
    for (i=0; i<VALUEWIDTH; i++) printf("─");
    printf("┘");
#else
    for (i=0; i<PRINTWIDTH; i++) printf("-");
#endif
    printf("\n\n");
}

void ghost_printLine(const char *label, const char *unit, const char *fmt, ...)
{
    va_list args;
    va_start(args,fmt);
    char dummy[1025];
    vsnprintf(dummy,1024,fmt,args);
    va_end(args);

#ifdef PRETTYPRINT
    printf("│");
#endif
    if (unit) {
        int unitLen = strlen(unit);
        printf("%-*s (%s)%s%*s",LABELWIDTH-unitLen-3,label,unit,PRINTSEP,VALUEWIDTH,dummy);
    } else {
        printf("%-*s%s%*s",LABELWIDTH,label,PRINTSEP,VALUEWIDTH,dummy);
    }
#ifdef PRETTYPRINT
    printf("│");
#endif
    printf("\n");
}



static char *env(char *key)
{
    int i=0;
    while (environ[i]) {
        if (!strncasecmp(key,environ[i],strlen(key)))
        {
            return environ[i]+strlen(key)+1;
        }
        i++;
    }
    return "undefined";

}

ghost_error_t ghost_printSysInfo()
{
    int nranks;
    int nnodes;
    int myrank;
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(MPI_COMM_WORLD,&nranks));
    GHOST_CALL_RETURN(ghost_getNumberOfNodes(MPI_COMM_WORLD,&nnodes));
    GHOST_CALL_RETURN(ghost_getRank(MPI_COMM_WORLD,&myrank));

#ifdef GHOST_HAVE_CUDA
    int cuVersion;
    GHOST_CALL_RETURN(ghost_cu_getVersion(&cuVersion));

    ghost_gpu_info_t * CUdevInfo;
    GHOST_CALL_RETURN(ghost_cu_getDeviceInfo(&CUdevInfo));
#endif
    if (myrank == 0) {

        int nthreads;
        int nphyscores;
        int ncores;
        ghost_getNumberOfPhysicalCores(&nphyscores);
        ghost_getNumberOfHwThreads(&ncores);

#ifdef GHOST_HAVE_OPENMP
        char omp_sched_str[32];
        omp_sched_t omp_sched;
        int omp_sched_mod;
        omp_get_schedule(&omp_sched,&omp_sched_mod);
        switch (omp_sched) {
            case omp_sched_static:
                sprintf(omp_sched_str,"static,%d",omp_sched_mod);
                break;
            case omp_sched_dynamic:
                sprintf(omp_sched_str,"dynamic,%d",omp_sched_mod);
                break;
            case omp_sched_guided:
                sprintf(omp_sched_str,"guided,%d",omp_sched_mod);
                break;
            case omp_sched_auto:
                sprintf(omp_sched_str,"auto,%d",omp_sched_mod);
                break;
            default:
                sprintf(omp_sched_str,"unknown");
                break;
        }
#else
        char omp_sched_str[] = "N/A";
#endif
#pragma omp parallel
#pragma omp master
        nthreads = ghost_ompGetNumThreads();

        uint64_t cacheSize;
        unsigned int cacheLineSize;
        ghost_getSizeOfLLC(&cacheSize);
        ghost_getSizeOfCacheLine(&cacheLineSize);

        ghost_printHeader("System");
        ghost_printLine("Overall nodes",NULL,"%d",nnodes);
        ghost_printLine("Overall MPI processes",NULL,"%d",nranks);
        ghost_printLine("MPI processes per node",NULL,"%d",nranks/nnodes);
        ghost_printLine("Avail. threads (phys/HW) per node",NULL,"%d/%d",nphyscores,ncores);
        ghost_printLine("OpenMP threads per node",NULL,"%d",nranks/nnodes*nthreads);
        ghost_printLine("OpenMP threads per process",NULL,"%d",nthreads);
        ghost_printLine("OpenMP scheduling",NULL,"%s",omp_sched_str);
        ghost_printLine("KMP_BLOCKTIME",NULL,"%s",env("KMP_BLOCKTIME"));
        ghost_printLine("LLC size","MiB","%.2f",cacheSize*1.0/(1024.*1024.));
        ghost_printLine("Cache line size","B","%.2f",cacheLineSize);
#ifdef GHOST_HAVE_CUDA
        ghost_printLine("CUDA version",NULL,"%d",cuVersion);
        ghost_printLine("CUDA devices",NULL,NULL);
        int j;
        for (j=0; j<CUdevInfo->nDistinctDevices; j++) {
            if (strcasecmp(CUdevInfo->names[j],"None")) {
                ghost_printLine("",NULL,"%dx %s",CUdevInfo->nDevices[j],CUdevInfo->names[j]);
            }
        }
#endif
        ghost_printFooter();
    }

    return GHOST_SUCCESS;

}

ghost_error_t ghost_printGhostInfo() 
{
    int myrank;
    GHOST_CALL_RETURN(ghost_getRank(MPI_COMM_WORLD,&myrank));

    if (myrank == 0) {
        
        ghost_printHeader("%s", GHOST_NAME);
        ghost_printLine("Version",NULL,"%s",GHOST_VERSION);
        ghost_printLine("Build date",NULL,"%s",__DATE__);
        ghost_printLine("Build time",NULL,"%s",__TIME__);
#ifdef MIC
        ghost_printLine("MIC kernels",NULL,"enabled");
#else
        ghost_printLine("MIC kernels",NULL,"disabled");
#endif
#ifdef AVX
        ghost_printLine("AVX kernels",NULL,"enabled");
#else
        ghost_printLine("AVX kernels",NULL,"disabled");
#endif
#ifdef SSE
        ghost_printLine("SSE kernels",NULL,"enabled");
#else
        ghost_printLine("SSE kernels",NULL,"disabled");
#endif
#ifdef GHOST_HAVE_OPENMP
        ghost_printLine("OpenMP support",NULL,"enabled");
#else
        ghost_printLine("OpenMP support",NULL,"disabled");
#endif
#ifdef GHOST_HAVE_MPI
        ghost_printLine("MPI support",NULL,"enabled");
#else
        ghost_printLine("MPI support",NULL,"disabled");
#endif
#ifdef GHOST_HAVE_CUDA
        ghost_printLine("CUDA support",NULL,"enabled");
#else
        ghost_printLine("CUDA support",NULL,"disabled");
#endif
#ifdef LIKWID
        ghost_printLine("Likwid support",NULL,"enabled");
        printf("Likwid support                   :      enabled\n");
#else
        ghost_printLine("Likwid support",NULL,"disabled");
#endif
        ghost_printFooter();
    }

    return GHOST_SUCCESS;
}


/*void ghost_freeCommunicator( ghost_comm_t* const comm ) 
  {
  if(comm) {
  free(comm->lnEnts);
  free(comm->lnrows);
  free(comm->lfEnt);
  free(comm->lfRow);
  free(comm->wishes);
  free(comm->dues);
  if (comm->wishlist)
  free(comm->wishlist[0]);
  if (comm->duelist)
  free(comm->duelist[0]);
  free(comm->wishlist);
  free(comm->duelist);
//free(comm->due_displ);
//free(comm->wish_displ);
free(comm->hput_pos);
free(comm);
}
}*/

char * ghost_modeName(int spmvmOptions) 
{
    if (spmvmOptions & GHOST_SPMVM_MODE_NOMPI)
        return "non-MPI";
    if (spmvmOptions & GHOST_SPMVM_MODE_VECTORMODE)
        return "vector mode";
    if (spmvmOptions & GHOST_SPMVM_MODE_GOODFAITH)
        return "g/f hybrid";
    if (spmvmOptions & GHOST_SPMVM_MODE_TASKMODE)
        return "task mode";
    return "invalid";

}

int ghost_symmetryValid(int symmetry)
{
    if ((symmetry & GHOST_BINCRS_SYMM_GENERAL) &&
            (symmetry & ~GHOST_BINCRS_SYMM_GENERAL))
        return 0;

    if ((symmetry & GHOST_BINCRS_SYMM_SYMMETRIC) &&
            (symmetry & ~GHOST_BINCRS_SYMM_SYMMETRIC))
        return 0;

    return 1;
}

char * ghost_symmetryName(int symmetry)
{
    if (symmetry & GHOST_BINCRS_SYMM_GENERAL)
        return "General";

    if (symmetry & GHOST_BINCRS_SYMM_SYMMETRIC)
        return "Symmetric";

    if (symmetry & GHOST_BINCRS_SYMM_SKEW_SYMMETRIC) {
        if (symmetry & GHOST_BINCRS_SYMM_HERMITIAN)
            return "Skew-hermitian";
        else
            return "Skew-symmetric";
    } else {
        if (symmetry & GHOST_BINCRS_SYMM_HERMITIAN)
            return "Hermitian";
    }

    return "Invalid";
}

int ghost_datatypeValid(int datatype)
{
    if ((datatype & GHOST_BINCRS_DT_FLOAT) &&
            (datatype & GHOST_BINCRS_DT_DOUBLE))
        return 0;

    if (!(datatype & GHOST_BINCRS_DT_FLOAT) &&
            !(datatype & GHOST_BINCRS_DT_DOUBLE))
        return 0;

    if ((datatype & GHOST_BINCRS_DT_REAL) &&
            (datatype & GHOST_BINCRS_DT_COMPLEX))
        return 0;

    if (!(datatype & GHOST_BINCRS_DT_REAL) &&
            !(datatype & GHOST_BINCRS_DT_COMPLEX))
        return 0;

    return 1;
}

char * ghost_datatypeName(int datatype)
{
    if (datatype & GHOST_BINCRS_DT_FLOAT) {
        if (datatype & GHOST_BINCRS_DT_REAL)
            return "float";
        else
            return "complex float";
    } else {
        if (datatype & GHOST_BINCRS_DT_REAL)
            return "double";
        else
            return "complex double";
    }
}

char * ghost_workdistName(int options)
{
    if (options & GHOST_CONTEXT_DIST_NZ)
        return "Equal no. of nonzeros";
    else
        return "Equal no. of rows";
}

ghost_midx_t ghost_pad(ghost_midx_t nrows, ghost_midx_t padding) 
{
    if (padding < 1 || nrows < 1) {
        return nrows;
    }
    
    ghost_midx_t nrowsPadded;

    if (nrows % padding != 0) {
        nrowsPadded = nrows + padding - nrows % padding;
    } else {
        nrowsPadded = nrows;
    }
    return nrowsPadded;
}
void *ghost_malloc(const size_t size)
{
    void *mem = NULL;

    if (size/(1024.*1024.*1024.) > 1.) {
        DEBUG_LOG(1,"Allocating big array of size %f GB",size/(1024.*1024.*1024.));
    }

    mem = malloc(size);

    if( ! mem ) {
        //  ABORT("Error in memory allocation of %zu bytes: %s",size,strerror(errno));
    }
    return mem;
}

void *ghost_malloc_align(const size_t size, const size_t align)
{
    void *mem = NULL;
    int ierr;

    if ((ierr = posix_memalign((void**) &mem, align, size)) != 0) {
        ERROR_LOG("Error while allocating using posix_memalign: %s",strerror(ierr));
    }

    return mem;
}


int ghost_dataTypeIdx(int datatype)
{
    if (datatype & GHOST_BINCRS_DT_FLOAT) {
        if (datatype & GHOST_BINCRS_DT_COMPLEX)
            return GHOST_DT_C_IDX;
        else
            return GHOST_DT_S_IDX;
    } else {
        if (datatype & GHOST_BINCRS_DT_COMPLEX)
            return GHOST_DT_Z_IDX;
        else
            return GHOST_DT_D_IDX;
    }
}





void ghost_ompSetNumThreads(int nthreads)
{
#ifdef GHOST_HAVE_OPENMP
    omp_set_num_threads(nthreads);
#else
    UNUSED(nthreads);
#endif
}

int ghost_ompGetNumThreads()
{
#ifdef GHOST_HAVE_OPENMP
    return omp_get_num_threads();
#else 
    return 1;
#endif
}

int ghost_ompGetThreadNum()
{
#ifdef GHOST_HAVE_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

int ghost_hash(int a, int b, int c)
{
    a -= b; a -= c; a ^= (c>>13);
    b -= c; b -= a; b ^= (a<<8);
    c -= a; c -= b; c ^= (b>>13);
    a -= b; a -= c; a ^= (c>>12);
    b -= c; b -= a; b ^= (a<<16);
    c -= a; c -= b; c ^= (b>>5);
    a -= b; a -= c; a ^= (c>>3);
    b -= c; b -= a; b ^= (a<<10);
    c -= a; c -= b; c ^= (b>>15);

    return c;
}


