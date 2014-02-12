#define _GNU_SOURCE
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/densemat.h"
#include "ghost/context.h"
#include "ghost/sparsemat.h"
#include "ghost/math.h"
#include "ghost/task.h"
#include "ghost/constants.h"
#include "ghost/locality.h"
#include "ghost/machine.h"
#include "ghost/io.h"
#include "ghost/log.h"
#include <libgen.h>
#include <unistd.h>
#include <byteswap.h>

#include <errno.h>
#ifdef GHOST_HAVE_OPENMP
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
#define HEADERHEIGHT 3
#define FOOTERHEIGHT 1

#ifdef PRETTYPRINT
#define PRINTSEP "┊"
#else
#define PRINTSEP ":"
#endif

#define VALUEWIDTH (PRINTWIDTH-LABELWIDTH-(int)strlen(PRINTSEP))


extern char ** environ;

void ghost_printHeader(char **str, const char *fmt, ...)
{
    size_t headerLen = (PRINTWIDTH+1)*HEADERHEIGHT+1;
    *str = realloc(*str,strlen(*str)+headerLen);
    memset(*str+strlen(*str),'\0',1);

    char label[1024] = "";

    if (fmt != NULL) {
        va_list args;
        va_start(args,fmt);
        vsnprintf(label,1024,fmt,args);
        va_end(args);
    }

    const int spacing = 4;
    int offset = 0;
    int len = strlen(label);
    int nDash = (PRINTWIDTH-2*spacing-len)/2;
    int rem = (PRINTWIDTH-2*spacing-len)%2;
    int i;
#ifdef PRETTYPRINT
    sprintf(*str,"┌");
    for (i=0; i<PRINTWIDTH-2; i++) sprintf(*str,"─");
    sprintf(*str,"┐");
    sprintf(*str,"\n");
    sprintf(*str,"├");
    for (i=0; i<nDash-1; i++) sprintf(*str,"─");
    for (i=0; i<spacing; i++) sprintf(*str," ");
    sprintf(*str,"%s",label);
    for (i=0; i<spacing+rem; i++) sprintf(*str," ");
    for (i=0; i<nDash-1; i++) sprintf(*str,"─");
    sprintf(*str,"┤");
    sprintf(*str,"\n");
    sprintf(*str,"├");
    for (i=0; i<LABELWIDTH; i++) sprintf(*str,"─");
    sprintf(*str,"┬");
    for (i=0; i<VALUEWIDTH; i++) sprintf(*str,"─");
    sprintf(*str,"┤");
    sprintf(*str,"\n");
#else
    for (i=0; i<PRINTWIDTH; i++) sprintf(*str+offset+i,"-");
    offset = PRINTWIDTH;
    sprintf(*str+offset,"\n");
    offset++;
    for (i=0; i<nDash; i++) sprintf(*str+offset+i,"-");
    offset += nDash;
    for (i=0; i<spacing; i++) sprintf(*str+offset+i," ");
    offset += spacing;
    sprintf(*str+offset,"%s",label);
    offset += strlen(label);
    for (i=0; i<spacing+rem; i++) sprintf(*str+offset+i," ");
    offset += spacing+rem;
    for (i=0; i<nDash; i++) sprintf(*str+offset+i,"-");
    offset += nDash;
    sprintf(*str+offset,"\n");
    offset++;
    for (i=0; i<PRINTWIDTH; i++) sprintf(*str+offset+i,"-");
    offset += PRINTWIDTH;
    sprintf(*str+offset,"\n");
#endif
}

void ghost_printFooter(char **str) 
{
    size_t len = strlen(*str);
    size_t footerLen = (PRINTWIDTH+1)*FOOTERHEIGHT+1;
    *str = realloc(*str,strlen(*str)+footerLen);
    int i;
#ifdef PRETTYPRINT
    sprintf(*str,"└");
    for (i=0; i<LABELWIDTH; i++) sprintf(*str,"─");
    sprintf(*str,"┴");
    for (i=0; i<VALUEWIDTH; i++) sprintf(*str,"─");
    sprintf(*str,"┘");
#else
    for (i=0; i<PRINTWIDTH; i++) sprintf(*str+len+i,"-");
#endif
    sprintf(*str+len+PRINTWIDTH,"\n");
}

void ghost_printLine(char **str, const char *label, const char *unit, const char *fmt, ...)
{
    size_t len = strlen(*str);

    *str = realloc(*str,len+PRINTWIDTH+2);
    memset(*str+len,'\0',1);

    va_list args;
    va_start(args,fmt);
    char dummy[1025];
    vsnprintf(dummy,1024,fmt,args);
    va_end(args);

#ifdef PRETTYPRINT
    sprintf(*str,"│");
#endif
    if (unit) {
        int unitLen = strlen(unit);
        sprintf(*str+len,"%-*s (%s)%s%*s",LABELWIDTH-unitLen-3,label,unit,PRINTSEP,VALUEWIDTH,dummy);
    } else {
        sprintf(*str+len,"%-*s%s%*s",LABELWIDTH,label,PRINTSEP,VALUEWIDTH,dummy);
    }
#ifdef PRETTYPRINT
    sprintf(*str+len,"│");
#endif
    sprintf(*str+len+PRINTWIDTH,"\n");
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


ghost_error_t ghost_sysInfoString(char **str)
{

    int nranks;
    int nnodes;

    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(MPI_COMM_WORLD,&nranks));
    GHOST_CALL_RETURN(ghost_getNumberOfNodes(MPI_COMM_WORLD,&nnodes));


#ifdef GHOST_HAVE_CUDA
    int cuVersion;
    GHOST_CALL_RETURN(ghost_cu_getVersion(&cuVersion));

    ghost_gpu_info_t * CUdevInfo;
    GHOST_CALL_RETURN(ghost_cu_getDeviceInfo(&CUdevInfo));
#endif


    int nthreads;
    int nphyscores;
    int ncores;
    ghost_getNumberOfCores(&nphyscores,GHOST_NUMANODE_ANY);
    ghost_getNumberOfPUs(&ncores,GHOST_NUMANODE_ANY);

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

    ghost_printHeader(str,"System");

    ghost_printLine(str,"Overall nodes",NULL,"%d",nnodes); 
    ghost_printLine(str,"Overall MPI processes",NULL,"%d",nranks);
    ghost_printLine(str,"MPI processes per node",NULL,"%d",nranks/nnodes);
    ghost_printLine(str,"Avail. threads (phys/HW) per node",NULL,"%d/%d",nphyscores,ncores);
    ghost_printLine(str,"OpenMP threads per node",NULL,"%d",nranks/nnodes*nthreads);
    ghost_printLine(str,"OpenMP threads per process",NULL,"%d",nthreads);
    ghost_printLine(str,"OpenMP scheduling",NULL,"%s",omp_sched_str);
    ghost_printLine(str,"KMP_BLOCKTIME",NULL,"%s",env("KMP_BLOCKTIME"));
    ghost_printLine(str,"LLC size","MiB","%.2f",cacheSize*1.0/(1024.*1024.));
    ghost_printLine(str,"Cache line size","B","%.2f",cacheLineSize);
#ifdef GHOST_HAVE_CUDA
    ghost_printLine(str,"CUDA version",NULL,"%d",cuVersion);
    ghost_printLine(str,"CUDA devices",NULL,NULL);
    int j;
    for (j=0; j<CUdevInfo->nDistinctDevices; j++) {
        if (strcasecmp(CUdevInfo->names[j],"None")) {
            ghost_printLine(str,"",NULL,"%dx %s",CUdevInfo->nDevices[j],CUdevInfo->names[j]);
        }
    }
#endif
    ghost_printFooter(str);

    return GHOST_SUCCESS;

}

ghost_error_t ghost_infoString(char **str) 
{
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);

    ghost_printHeader(str,"%s", GHOST_NAME); 
    ghost_printLine(str,"Version",NULL,"%s",GHOST_VERSION);
    ghost_printLine(str,"Build date",NULL,"%s",__DATE__);
    ghost_printLine(str,"Build time",NULL,"%s",__TIME__);
#ifdef GHOST_HAVE_MIC
    ghost_printLine(str,"MIC kernels",NULL,"Enabled");
#else
    ghost_printLine(str,"MIC kernels",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_AVX
    ghost_printLine(str,"AVX kernels",NULL,"Enabled");
#else
    ghost_printLine(str,"AVX kernels",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_SSE
    ghost_printLine(str,"SSE kernels",NULL,"Enabled");
#else
    ghost_printLine(str,"SSE kernels",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_OPENMP
    ghost_printLine(str,"OpenMP support",NULL,"Enabled");
#else
    ghost_printLine(str,"OpenMP support",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_MPI
    ghost_printLine(str,"MPI support",NULL,"Enabled");
#else
    ghost_printLine(str,"MPI support",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_CUDA
    ghost_printLine(str,"CUDA support",NULL,"Enabled");
#else
    ghost_printLine(str,"CUDA support",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_INSTR_LIKWID
    ghost_printLine(str,"Instrumentation",NULL,"Likwid");
#elif defined(GHOST_HAVE_INSTR_TIMING)
    ghost_printLine(str,"Instrumentation",NULL,"Timing");
#else
    ghost_printLine(str,"Instrumentation",NULL,"Disabled");
#endif
    ghost_printFooter(str);

    return GHOST_SUCCESS;

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

ghost_error_t ghost_malloc(void **mem, const size_t size)
{
    if (!mem) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    if (size/(1024.*1024.*1024.) > 1.) {
        DEBUG_LOG(1,"Allocating big array of size %f GB",size/(1024.*1024.*1024.));
    }

    *mem = malloc(size);

    if(!(*mem)) {
        ERROR_LOG("Malloc failed: %s",strerror(errno));
        return GHOST_ERR_UNKNOWN;
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_malloc_align(void **mem, const size_t size, const size_t align)
{
    int ierr;

    if ((ierr = posix_memalign(mem, align, size)) != 0) {
        ERROR_LOG("Error while allocating using posix_memalign: %s",strerror(ierr));
        return GHOST_ERR_UNKNOWN;
    }

    return GHOST_SUCCESS;
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


