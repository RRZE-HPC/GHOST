#define _GNU_SOURCE
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/vec.h"
#include "ghost/context.h"
#include "ghost/mat.h"
#include "ghost/math.h"
#include "ghost/taskq.h"
#include "ghost/constants.h"
#include "ghost/affinity.h"
#include "ghost/io.h"
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

#ifdef GHOST_HAVE_MPI
static int MPIwasInitialized;
MPI_Datatype GHOST_MPI_DT_C;
MPI_Op GHOST_MPI_OP_SUM_C;
MPI_Datatype GHOST_MPI_DT_Z;
MPI_Op GHOST_MPI_OP_SUM_Z;
MPI_Comm ghost_node_comm = MPI_COMM_NULL;
int ghost_node_rank = -1;
#else
int ghost_node_comm = 0;
int ghost_node_rank = 0;
#endif
hwloc_topology_t topology;

ghost_type_t ghost_type = GHOST_TYPE_INVALID;
ghost_hybridmode_t ghost_hybridmode = GHOST_HYBRIDMODE_INVALID;
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
        va_list args;
        va_start(args,fmt);
        char label[1024];
        vsnprintf(label,1024,fmt,args);
        va_end(args);

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

void ghost_printMatrixInfo(ghost_mat_t *mat)
{
    ghost_midx_t nrows = ghost_getMatNrows(mat);
    ghost_midx_t nnz = ghost_getMatNnz(mat);
    
    int myrank = ghost_getRank(mat->context->mpicomm);;

    if (myrank == 0) {

    char *matrixLocation;
    if (mat->traits->flags & GHOST_SPM_DEVICE)
        matrixLocation = "Device";
    else if (mat->traits->flags & GHOST_SPM_HOST)
        matrixLocation = "Host";
    else
        matrixLocation = "Default";


    ghost_printHeader(mat->name);
    ghost_printLine("Data type",NULL,"%s",ghost_datatypeName(mat->traits->datatype));
    ghost_printLine("Matrix location",NULL,"%s",matrixLocation);
    ghost_printLine("Number of rows",NULL,"%"PRmatIDX,nrows);
    ghost_printLine("Number of nonzeros",NULL,"%"PRmatNNZ,nnz);
    ghost_printLine("Avg. nonzeros per row",NULL,"%.3f",(double)nnz/nrows);

    ghost_printLine("Full   matrix format",NULL,"%s",mat->formatName(mat));
    if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED)
    {
        ghost_printLine("Local  matrix format",NULL,"%s",mat->localPart->formatName(mat->localPart));
        ghost_printLine("Remote matrix format",NULL,"%s",mat->remotePart->formatName(mat->remotePart));
        ghost_printLine("Local  matrix symmetry",NULL,"%s",ghost_symmetryName(mat->localPart->traits->symmetry));
    } else {
        ghost_printLine("Full   matrix symmetry",NULL,"%s",ghost_symmetryName(mat->traits->symmetry));
    }

    ghost_printLine("Full   matrix size (rank 0)","MB","%u",mat->byteSize(mat)/(1024*1024));
    if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED)
    {
        ghost_printLine("Local  matrix size (rank 0)","MB","%u",mat->localPart->byteSize(mat->localPart)/(1024*1024));
        ghost_printLine("Remote matrix size (rank 0)","MB","%u",mat->remotePart->byteSize(mat->remotePart)/(1024*1024));
    }

    mat->printInfo(mat);
    ghost_printFooter();

    }

}

void ghost_printContextInfo(ghost_context_t *context)
{
    int nranks = ghost_getNumberOfPhysicalCores(context->mpicomm);
    int myrank = ghost_getRank(context->mpicomm);

        if (myrank == 0) {
        char *contextType = "";
        if (context->flags & GHOST_CONTEXT_DISTRIBUTED)
            contextType = "Distributed";
        else if (context->flags & GHOST_CONTEXT_GLOBAL)
            contextType = "Global";


        ghost_printHeader("Context");
        ghost_printLine("MPI processes",NULL,"%d",nranks);
        ghost_printLine("Number of rows",NULL,"%"PRmatIDX,context->gnrows);
        ghost_printLine("Type",NULL,"%s",contextType);
        ghost_printLine("Work distribution scheme",NULL,"%s",ghost_workdistName(context->flags));
        ghost_printFooter();
    }

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

void ghost_printSysInfo()
{
    int nproc = ghost_getNumberOfRanks(MPI_COMM_WORLD);
    int nnodes = ghost_getNumberOfNodes();

#ifdef GHOST_HAVE_CUDA
    ghost_acc_info_t * CUdevInfo = CU_getDeviceInfo();
#endif
    if (ghost_getRank(MPI_COMM_WORLD) == 0) {

        int nthreads;
        int nphyscores = ghost_getNumberOfPhysicalCores();
        int ncores = ghost_getNumberOfHwThreads();

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

        ghost_printHeader("System");
        ghost_printLine("Overall nodes",NULL,"%d",nnodes);
        ghost_printLine("Overall MPI processes",NULL,"%d",nproc);
        ghost_printLine("MPI processes per node",NULL,"%d",nproc/nnodes);
        ghost_printLine("Avail. threads (phys/HW) per node",NULL,"%d/%d",nphyscores,ncores);
        ghost_printLine("OpenMP threads per node",NULL,"%d",nproc/nnodes*nthreads);
        ghost_printLine("OpenMP threads per process",NULL,"%d",nthreads);
        ghost_printLine("OpenMP scheduling",NULL,"%s",omp_sched_str);
        ghost_printLine("KMP_BLOCKTIME",NULL,"%s",env("KMP_BLOCKTIME"));
        ghost_printLine("LLC size","MiB","%.2f",ghost_getSizeOfLLC()*1.0/(1024.*1024.));
#ifdef GHOST_HAVE_CUDA
        ghost_printLine("CUDA version",NULL,"%s",CU_getVersion());
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

}

void ghost_printGhostInfo() 
{

    if (ghost_getRank(MPI_COMM_WORLD)==0) {


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
}

void ghost_referenceSolver(ghost_vec_t *nodeLHS, char *matrixPath, int datatype, ghost_vec_t *rhs, int nIter, int spmvmOptions)
{

    DEBUG_LOG(1,"Computing reference solution");
    int me = ghost_getRank(nodeLHS->context->mpicomm);
    
    char *zero = (char *)ghost_malloc(ghost_sizeofDataType(datatype));
    memset(zero,0,ghost_sizeofDataType(datatype));
    ghost_vec_t *globLHS; 
    ghost_mtraits_t trait = {.format = GHOST_SPM_FORMAT_CRS, .flags = GHOST_SPM_HOST, .aux = NULL, .datatype = datatype};
    ghost_context_t *context;

    ghost_matfile_header_t fileheader;
    ghost_readMatFileHeader(matrixPath,&fileheader);

    ghost_createContext(&context,GHOST_GET_DIM_FROM_MATRIX,GHOST_GET_DIM_FROM_MATRIX,GHOST_CONTEXT_GLOBAL,matrixPath,MPI_COMM_WORLD,1.0);
    ghost_mat_t *mat = ghost_createMatrix(context, &trait, 1);
    mat->fromFile(mat,matrixPath);
    ghost_vtraits_t rtraits = GHOST_VTRAITS_INITIALIZER;
    rtraits.flags = GHOST_VEC_RHS|GHOST_VEC_HOST;
    rtraits.datatype = rhs->traits->datatype;;
       rtraits.nvecs=rhs->traits->nvecs;

    ghost_vec_t *globRHS = ghost_createVector(context, &rtraits);
    globRHS->fromScalar(globRHS,zero);


    DEBUG_LOG(2,"Collection RHS vector for reference solver");
    rhs->collect(rhs,globRHS);

    if (me==0) {
        DEBUG_LOG(1,"Computing actual reference solution with one process");


        ghost_vtraits_t ltraits = GHOST_VTRAITS_INITIALIZER;
        ltraits.flags = GHOST_VEC_LHS|GHOST_VEC_HOST;
        ltraits.datatype = rhs->traits->datatype;
        ltraits.nvecs = rhs->traits->nvecs;

        globLHS = ghost_createVector(context, &ltraits); 
        globLHS->fromScalar(globLHS,&zero);

        int iter;

        if (mat->traits->symmetry == GHOST_BINCRS_SYMM_GENERAL) {
            for (iter=0; iter<nIter; iter++) {
                mat->spmv(mat,globLHS,globRHS,spmvmOptions);
            }
        } else if (mat->traits->symmetry == GHOST_BINCRS_SYMM_SYMMETRIC) {
            WARNING_LOG("Computing the refernce solution for a symmetric matrix is not implemented!");
            for (iter=0; iter<nIter; iter++) {
            }
        }

        globRHS->destroy(globRHS);
        ghost_freeContext(context);
    } else {
        ghost_vtraits_t ltraits = GHOST_VTRAITS_INITIALIZER;
        ltraits.flags = GHOST_VEC_LHS|GHOST_VEC_HOST|GHOST_VEC_DUMMY;
        ltraits.datatype = rhs->traits->datatype;
        ltraits.nvecs = rhs->traits->nvecs;
        globLHS = ghost_createVector(context, &ltraits);
    }
    DEBUG_LOG(1,"Scattering result of reference solution");

    nodeLHS->fromScalar(nodeLHS,&zero);
    globLHS->distribute(globLHS, nodeLHS);

    globLHS->destroy(globLHS);
    mat->destroy(mat);
    

    free(zero);
    DEBUG_LOG(1,"Reference solution has been computed and scattered successfully");
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
    if (options & GHOST_CONTEXT_WORKDIST_NZE)
        return "Equal no. of nonzeros";
    else
        return "Equal no. of rows";
}

size_t ghost_sizeofDataType(int dt)
{
    size_t size;

    if (dt & GHOST_BINCRS_DT_FLOAT)
        size = sizeof(float);
    else
        size = sizeof(double);

    if (dt & GHOST_BINCRS_DT_COMPLEX)
        size *= 2;

    return size;
}

int ghost_pad(int nrows, int padding) 
{
    int nrowsPadded;

    if(  nrows % padding != 0) {
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
        ABORT("Error while allocating using posix_memalign: %s",strerror(ierr));
    }

    return mem;
}

double ghost_bench_spmvm(ghost_context_t *context, ghost_vec_t *res, ghost_mat_t *mat, ghost_vec_t *invec, 
        int *spmvmOptions, int nIter)
{
    DEBUG_LOG(1,"Benchmarking the SpMVM");
    int it;
    double time = 0;
//    double ttime = 0;
    double oldtime=1e9;
    //struct timespec end,start;

    ghost_spmvsolver_t solver = NULL;

    ghost_pickSpMVMMode(context,spmvmOptions);
    solver = context->spmvsolvers[ghost_getSpmvmModeIdx(*spmvmOptions)];

    if (!solver) {
        DEBUG_LOG(1,"The solver for the specified is not available, skipping");
        return -1.0;
    }

#ifdef GHOST_HAVE_MPI
    MPI_safecall(MPI_Barrier(context->mpicomm));
#endif

//    ttime = ghost_wctime();
    for( it = 0; it < nIter; it++ ) {
        //clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
        time = ghost_wctime();
        solver(context,res,mat,invec,*spmvmOptions);

#ifdef GHOST_HAVE_CUDA
        CU_barrier();
#endif
#ifdef GHOST_HAVE_MPI
        MPI_safecall(MPI_Barrier(context->mpicomm));
#endif
        //clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
        //time = ghost_timediff(start,end);
        time = ghost_wctime()-time;
    //    printf("%f\n",time);
//        if (time < 0)
//            printf("dummy\n");
        time = time<oldtime?time:oldtime;
        oldtime=time;
    }
    //DEBUG_LOG(0,"Total time: %f sec",ghost_wctime()-ttime);
    solver(NULL,NULL,NULL,NULL,0); // clean up

    DEBUG_LOG(1,"Downloading result from device");
    res->download(res);

    if ( *spmvmOptions & GHOST_SPMVM_MODES_COMBINED)  {
        res->permute(res,res->context->invRowPerm);
    } else if ( *spmvmOptions & GHOST_SPMVM_MODES_SPLIT ) {
        // one of those must return immediately
        res->permute(res,res->context->invRowPerm);
        res->permute(res,res->context->invRowPerm);
    }

    return time;
}

void ghost_pickSpMVMMode(ghost_context_t * context, int *spmvmOptions)
{
    if (!(*spmvmOptions & GHOST_SPMVM_MODES_ALL)) { // no mode specified
#ifdef GHOST_HAVE_MPI
        if (context->flags & GHOST_CONTEXT_GLOBAL)
            *spmvmOptions |= GHOST_SPMVM_MODE_NOMPI;
        else
            *spmvmOptions |= GHOST_SPMVM_MODE_GOODFAITH;
#else
        UNUSED(context);
        *spmvmOptions |= GHOST_SPMVM_MODE_NOMPI;
#endif
        DEBUG_LOG(1,"No spMVM mode has been specified, picking a sensible default, namely %s",ghost_modeName(*spmvmOptions));

    }

}

int ghost_getSpmvmModeIdx(int spmvmOptions)
{
    if (spmvmOptions & GHOST_SPMVM_MODE_NOMPI)
        return GHOST_SPMVM_MODE_NOMPI_IDX;
    if (spmvmOptions & GHOST_SPMVM_MODE_VECTORMODE)
        return GHOST_SPMVM_MODE_VECTORMODE_IDX;
    if (spmvmOptions & GHOST_SPMVM_MODE_GOODFAITH)
        return GHOST_SPMVM_MODE_GOODFAITH_IDX;
    if (spmvmOptions & GHOST_SPMVM_MODE_TASKMODE)
        return GHOST_SPMVM_MODE_TASKMODE_IDX;

    return 0;
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


int ghost_archIsBigEndian()
{
    int test = 1;
    unsigned char *endiantest = (unsigned char *)&test;

    return (endiantest[0] == 0);
}


char ghost_datatypePrefix(int dt)
{
    char p;

    if (dt & GHOST_BINCRS_DT_FLOAT) {
        if (dt & GHOST_BINCRS_DT_COMPLEX)
            p = 'c';
        else
            p = 's';
    } else {
        if (dt & GHOST_BINCRS_DT_COMPLEX)
            p = 'z';
        else
            p = 'd';
    }

    return p;
}


ghost_midx_t ghost_globalIndex(ghost_context_t *ctx, ghost_midx_t lidx)
{
    if (ctx->flags & GHOST_CONTEXT_DISTRIBUTED)
        return ctx->lfRow[ghost_getRank(ctx->mpicomm)] + lidx;

    return lidx;    
}

int ghost_flopsPerSpmvm(int m_t, int v_t)
{
    int flops = 2;

    if (m_t & GHOST_BINCRS_DT_COMPLEX) {
        if (v_t & GHOST_BINCRS_DT_COMPLEX) {
            flops = 8;
        }
    } else {
        if (v_t & GHOST_BINCRS_DT_COMPLEX) {
            flops = 4;
        }
    }

    return flops;
}

ghost_vtraits_t * ghost_cloneVtraits(ghost_vtraits_t *t1)
{
    ghost_vtraits_t *t2 = (ghost_vtraits_t *)ghost_malloc(sizeof(ghost_vtraits_t));
    memcpy(t2,t1,sizeof(ghost_vtraits_t));

    return t2;
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

static unsigned int* ghost_rand_states=NULL;

unsigned int* ghost_getRandState()
{
        return &ghost_rand_states[ghost_ompGetThreadNum()];
}

void ghost_rand_init()
{
   int N_Th = 1;
#pragma omp parallel
{
  #pragma omp single
    N_Th = ghost_ompGetNumThreads();
}
     
    if( ghost_rand_states == NULL )    ghost_rand_states=(unsigned int*)malloc(N_Th*sizeof(unsigned int));
#pragma omp parallel
    {
        unsigned int seed=(unsigned int)ghost_hash(
                (int)ghost_wctimemilli(),
                (int)ghost_getRank(MPI_COMM_WORLD),
                (int)ghost_ompGetThreadNum());
        *ghost_getRandState()=seed;
    }
}


int ghost_init(int argc, char **argv)
{
#ifdef GHOST_HAVE_MPI
    int req, prov;

#ifdef GHOST_HAVE_OPENMP
    req = MPI_THREAD_MULTIPLE; 
#else
    req = MPI_THREAD_SINGLE;
#endif

    MPI_safecall(MPI_Initialized(&MPIwasInitialized));
    if (!MPIwasInitialized) {
        MPI_safecall(MPI_Init_thread(&argc, &argv, req, &prov ));

        if (req != prov) {
            WARNING_LOG("Required MPI threading level (%d) is not "
                    "provided (%d)!",req,prov);
        }
    } else {
        WARNING_LOG("MPI was already initialized, not doing it!");
    }

    MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&GHOST_MPI_DT_C));
    MPI_safecall(MPI_Type_commit(&GHOST_MPI_DT_C));
    MPI_safecall(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_c,1,&GHOST_MPI_OP_SUM_C));

    MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&GHOST_MPI_DT_Z));
    MPI_safecall(MPI_Type_commit(&GHOST_MPI_DT_Z));
    MPI_safecall(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_z,1,&GHOST_MPI_OP_SUM_Z));
    
    ghost_setupNodeMPI(MPI_COMM_WORLD);

#else // ifdef GHOST_HAVE_MPI
    UNUSED(argc);
    UNUSED(argv);

#endif // ifdef GHOST_HAVE_MPI

#if GHOST_HAVE_INSTR_LIKWID
    LIKWID_MARKER_INIT;

#pragma omp parallel
    LIKWID_MARKER_THREADINIT;
#endif

    
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);


    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology,cpuset,HWLOC_CPUBIND_PROCESS);
    if (hwloc_bitmap_weight(cpuset) < hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_PU)) {
        WARNING_LOG("GHOST is running in a restricted CPU set. This is probably not what you want because GHOST cares for pinning itself...");
    }
    hwloc_bitmap_free(cpuset);
 

    // auto-set rank types 
    int nnoderanks = ghost_getNumberOfRanks(ghost_node_comm);
    int noderank = ghost_getRank(ghost_node_comm);
    
    int ncudadevs = 0;
    int ndomains = 0;
    int nnumanodes = hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_NODE);
    ndomains += nnumanodes;
#if GHOST_HAVE_CUDA
    CU_getDeviceCount(&ncudadevs);
#endif
    ndomains += ncudadevs;


    if (ghost_type == GHOST_TYPE_INVALID) {
        if (noderank == 0) {
            ghost_setType(GHOST_TYPE_COMPUTE);
        } else if (noderank <= ncudadevs) {
            ghost_setType(GHOST_TYPE_CUDAMGMT);
        } else {
            ghost_setType(GHOST_TYPE_COMPUTE);
        }
    } 

#ifndef GHOST_HAVE_CUDA
    if (ghost_type == GHOST_TYPE_CUDAMGMT) {
        WARNING_LOG("This rank is supposed to be a CUDA management rank but CUDA is not available. Re-setting GHOST type");
        ghost_setType(GHOST_TYPE_COMPUTE);
    }
#endif


    int nLocalCompute = ghost_type==GHOST_TYPE_COMPUTE;
    int nLocalCuda = ghost_type==GHOST_TYPE_CUDAMGMT;

    int i;
    int localTypes[nnoderanks];

    for (i=0; i<ghost_getNumberOfRanks(ghost_node_comm); i++) {
        localTypes[i] = GHOST_TYPE_INVALID;
    }
    localTypes[noderank] = ghost_type;
#if GHOST_HAVE_MPI
    MPI_safecall(MPI_Allreduce(MPI_IN_PLACE,&nLocalCompute,1,MPI_INT,MPI_SUM,ghost_node_comm));
    MPI_safecall(MPI_Allreduce(MPI_IN_PLACE,&nLocalCuda,1,MPI_INT,MPI_SUM,ghost_node_comm));

#ifdef GHOST_HAVE_CUDA
    if (ncudadevs < nLocalCuda) {
        WARNING_LOG("There are %d CUDA management ranks on this node but only %d CUDA devices.",nLocalCuda,ncudadevs);
    }
#endif


    MPI_safecall(MPI_Allreduce(MPI_IN_PLACE,&localTypes,ghost_getNumberOfRanks(ghost_node_comm),MPI_INT,MPI_MAX,ghost_node_comm));
#endif   
    
    if (ghost_hybridmode == GHOST_HYBRIDMODE_INVALID) {
    if (nnoderanks <=  nLocalCuda+1) {
        ghost_hybridmode = GHOST_HYBRIDMODE_ONEPERNODE;
        INFO_LOG("One CPU rank per node");
    } else if (nnoderanks == nLocalCuda+nnumanodes) {
        ghost_hybridmode = GHOST_HYBRIDMODE_ONEPERNUMA;
        INFO_LOG("One CPU rank per NUMA domain");
    } else if (nnoderanks == hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_CORE)) {
        ghost_hybridmode = GHOST_HYBRIDMODE_ONEPERCORE;
        WARNING_LOG("One MPI process per core not supported");
    } else {
        WARNING_LOG("Invalid number of ranks on node");
    }
    }

    hwloc_cpuset_t mycpuset = hwloc_bitmap_alloc();
    hwloc_cpuset_t globcpuset = hwloc_bitmap_alloc();

    globcpuset = hwloc_bitmap_dup(hwloc_get_obj_by_depth(topology,HWLOC_OBJ_SYSTEM,0)->cpuset);

    hwloc_obj_t obj;
    ghost_hw_config_t hwconfig;
    ghost_getHwConfig(&hwconfig);

    if (hwconfig.maxCores == GHOST_HW_CONFIG_INVALID) {
        hwconfig.maxCores = ghost_getNumberOfPhysicalCores();
    }
    if (hwconfig.smtLevel == GHOST_HW_CONFIG_INVALID) {
        hwconfig.smtLevel = ghost_getSMTlevel();
    }
    ghost_setHwConfig(hwconfig);

    int cpu;
    hwloc_bitmap_foreach_begin(cpu,globcpuset);
    obj = hwloc_get_pu_obj_by_os_index(topology,cpu);
    if (obj->sibling_rank >= hwconfig.smtLevel) {
        hwloc_bitmap_clr(globcpuset,cpu);
    }
    if (obj->parent->logical_index >= hwconfig.maxCores) { 
        hwloc_bitmap_clr(globcpuset,cpu);
    }
    hwloc_bitmap_foreach_end();


#if GHOST_HAVE_CUDA
    int cudaDevice = 0;

    for (i=0; i<ghost_getNumberOfRanks(ghost_node_comm); i++) {
        if (localTypes[i] == GHOST_TYPE_CUDAMGMT) {
            if (i == ghost_getRank(ghost_node_comm)) {
                ghost_CUDA_init(cudaDevice);
            }
            cudaDevice++;
        }
    }


    // CUDA ranks have a physical core
    cudaDevice = 0;
    for (i=0; i<ghost_getNumberOfRanks(ghost_node_comm); i++) {
        if (localTypes[i] == GHOST_TYPE_CUDAMGMT) {
            hwloc_obj_t mynode = hwloc_get_obj_by_type(topology,HWLOC_OBJ_NODE,cudaDevice%nnumanodes);
            hwloc_obj_t runner = mynode;
            while (hwloc_compare_types(runner->type, HWLOC_OBJ_CORE) < 0) {
                runner = runner->first_child;
                char *foo;
                hwloc_bitmap_list_asprintf(&foo,runner->cpuset);
            }
            if (i == ghost_getRank(ghost_node_comm)) {
                hwloc_bitmap_copy(mycpuset,runner->cpuset);
            //    corestaken[runner->logical_index] = 1;
            }
            cudaDevice++;

            // delete CUDA cores from global cpuset
            hwloc_bitmap_andnot(globcpuset,globcpuset,runner->cpuset);
        }
    }
#endif
    int oversubscribed = 0;

    if (ghost_hybridmode == GHOST_HYBRIDMODE_ONEPERNODE) {
        if (ghost_type == GHOST_TYPE_COMPUTE) {
            hwloc_bitmap_copy(mycpuset,globcpuset);
        }
        hwloc_bitmap_andnot(globcpuset,globcpuset,globcpuset);
    } else if (ghost_hybridmode == GHOST_HYBRIDMODE_ONEPERNUMA) {
        int numaNode = 0;
        for (i=0; i<ghost_getNumberOfRanks(ghost_node_comm); i++) {
            if (localTypes[i] == GHOST_TYPE_COMPUTE) {
                if (hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_NODE) > numaNode) {
                    if (i == ghost_getRank(ghost_node_comm)) {
                        hwloc_bitmap_and(mycpuset,globcpuset,hwloc_get_obj_by_type(topology,HWLOC_OBJ_NODE,numaNode)->cpuset);
                    }
                    hwloc_bitmap_andnot(globcpuset,globcpuset,hwloc_get_obj_by_type(topology,HWLOC_OBJ_NODE,numaNode)->cpuset);
                    numaNode++;
                } else {
                    oversubscribed = 1;
                    WARNING_LOG("More processes than NUMA nodes");
                    break;
                }
            }
        }
    } 

    if (oversubscribed) {
        mycpuset = hwloc_bitmap_dup(hwloc_get_obj_by_depth(topology,HWLOC_OBJ_SYSTEM,0)->cpuset);
    }


    char *cpusetstr, *mycpusetstr;
    hwloc_bitmap_list_asprintf(&cpusetstr,mycpuset);
    INFO_LOG("Process cpuset (OS indexing): %s",cpusetstr);
    if (hwloc_bitmap_weight(globcpuset) > 0) {
        WARNING_LOG("There are unassigned cores");
    }
    ghost_thpool_init(mycpuset);
     
    ghost_rand_init();
     
    hwloc_bitmap_free(mycpuset);   
    hwloc_bitmap_free(globcpuset);   
    return GHOST_SUCCESS;
}

void ghost_finish()
{

    ghost_taskq_finish();
    ghost_thpool_finish();
    hwloc_topology_destroy(topology);
    
    free(ghost_rand_states);
    ghost_rand_states=NULL;

#if GHOST_HAVE_INSTR_LIKWID
    LIKWID_MARKER_CLOSE;
#endif


#if GHOST_HAVE_MPI
    MPI_safecall(MPI_Type_free(&GHOST_MPI_DT_C));
    MPI_safecall(MPI_Type_free(&GHOST_MPI_DT_Z));
    if (!MPIwasInitialized) {
        MPI_Finalize();
    }
#endif

}

size_t ghost_getSizeOfLLC()
{
    hwloc_obj_t obj;
    int depth;
    size_t size = 0;

    for (depth=0; depth<(int)hwloc_topology_get_depth(topology); depth++) {
        obj = hwloc_get_obj_by_depth(topology,depth,0);
        if (obj->type == HWLOC_OBJ_CACHE) {
            size = obj->attr->cache.size;
            break;
        }
    }
#if GHOST_HAVE_MIC
    size = size*ghost_getNumberOfPhysicalCores(); // the cache is shared but not reported so
#endif
    return size;
}
    
int ghost_setType(ghost_type_t t)
{
    ghost_type = t;

    return GHOST_SUCCESS;
}

int ghost_setHybridMode(ghost_hybridmode_t hm)
{
    ghost_hybridmode = hm;

    return GHOST_SUCCESS;
}

// http://burtleburtle.net/bob/hash/doobs.html
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
