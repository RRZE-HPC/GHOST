#include <ghost.h>
#include <ghost_vec.h>
#include <ghost_util.h>
#include <ghost_taskq.h>

#ifdef VT
#include <VT.h>
#endif
#include <omp.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <libgen.h>
#include <strings.h>
#ifdef GHOST_MPI
#include <mpi.h>
#endif

#define TASKING
#define CHECK // compare with reference solution

GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)
#define EPS 1.e-3

#ifdef TASKING
typedef struct {
	ghost_mat_t *mat;
	ghost_vec_t **lhs, **rhs;
	ghost_vtraits_t *ltr, *rtr;
	ghost_context_t *ctx;
	char *matfile;
	vecdt_t *lhsInit;
	void (*rhsInit)(int,int,void*);
} createDataArgs;

typedef struct {
	ghost_context_t *ctx;
	ghost_mat_t *mat;
	ghost_vec_t *lhs, *rhs;
	int *spmvmOptions;
	int nIter;
	double *time;
} benchArgs;


static void *createDataFunc(void *vargs)
{
	createDataArgs *args = (createDataArgs *)vargs;
	args->mat->fromFile(args->mat,args->matfile);
	*(args->lhs) = ghost_createVector(args->ctx,args->ltr);
	*(args->rhs) = ghost_createVector(args->ctx,args->rtr);
	(*(args->lhs))->fromScalar(*(args->lhs),args->lhsInit);
	(*(args->rhs))->fromFunc(*(args->rhs),args->rhsInit);

	return NULL;
}
static void *benchFunc(void *vargs)
{
//#pragma omp parallel
//	WARNING_LOG("Thread %d running @ core %d",omp_get_thread_num(),ghost_getCore());
	benchArgs *args = (benchArgs *)vargs;
	*(args->time) = ghost_bench_spmvm(args->ctx,args->lhs,args->mat,args->rhs,args->spmvmOptions,args->nIter);

	return NULL;
}
#endif

static void rhsVal (int i, int v, void *val) 
{
	UNUSED(i);
	UNUSED(v);
	*(vecdt_t *)val = 1+I*1;//i + (vecdt_t)1.0 + I*i;
}

int main( int argc, char* argv[] ) 
{

	int  mode, nIter = 50;
	double time;
	vecdt_t zero = 0.;
	matdt_t shift = 0.;

#ifdef CHECK
	ghost_midx_t i, errcount = 0;
	double mytol;
#endif

	int modes[] = {GHOST_SPMVM_MODE_NOMPI,
		/*GHOST_SPMVM_MODE_VECTORMODE,*/
		GHOST_SPMVM_MODE_GOODFAITH/*,
		GHOST_SPMVM_MODE_TASKMODE*/};
		int nModes = sizeof(modes)/sizeof(int);

	int spmvmOptions = GHOST_SPMVM_AXPY /* | GHOST_SPMVM_APPLY_SHIFT*/;

	ghost_mat_t *mat; // matrix
	ghost_vec_t *lhs = NULL; // lhs vector
	ghost_vec_t *rhs = NULL; // rhs vector

	ghost_context_t *context;

	char *matrixPath = argv[1];
	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = GHOST_SPM_FORMAT_CRS, .datatype = matdt, .shift = &shift);
	ghost_vtraits_t lvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .datatype = vecdt);
	ghost_vtraits_t rvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS, .datatype = vecdt);

	if (argc == 5) {
		if (!(strcasecmp(argv[2],"CRS")))
			mtraits.format = GHOST_SPM_FORMAT_CRS;
		else if (!(strcasecmp(argv[2],"SELL")))
			mtraits.format = GHOST_SPM_FORMAT_SELL;
		mtraits.flags = atoi(argv[3]);
		int sortBlock = atoi(argv[4]);
		int aux[2];
		aux[0] = sortBlock;
		//aux[1] = GHOST_SELL_CHUNKHEIGHT_ELLPACK; 
		aux[1] = 32; 
		mtraits.aux = &aux;
	}

	ghost_init(argc,argv);       // basic initialization

#ifndef TASKING
	ghost_pinThreads(GHOST_PIN_PHYS,NULL);
#endif

	context = ghost_createContext(GHOST_GET_DIM_FROM_MATRIX,GHOST_GET_DIM_FROM_MATRIX,GHOST_CONTEXT_DEFAULT,matrixPath,MPI_COMM_WORLD,1.0);
	mat = ghost_createMatrix(context,&mtraits,1);


#ifdef TASKING
//	ghost_setCore(23);
//#pragma omp parallel
//	WARNING_LOG("Main thread %d running @ core %d",omp_get_thread_num(),ghost_getCore());
	createDataArgs args = {.mat = mat, .lhs = &lhs, .rhs = &rhs, .matfile = matrixPath, .lhsInit = &zero, .rhsInit = rhsVal, .rtr = &rvtraits, .ltr = &lvtraits, .ctx = context};
	ghost_task_t *createDataTask = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &createDataFunc, &args, GHOST_TASK_DEFAULT);
	ghost_task_add(createDataTask);
#else
	mat->fromFile(mat,matrixPath);
	lhs = ghost_createVector(context,&lvtraits);
	rhs = ghost_createVector(context,&rvtraits);
	lhs->fromScalar(lhs,&zero);
	rhs->fromFunc(rhs,&rhsVal);
#endif

	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printContextInfo(context);

#ifdef TASKING
	ghost_task_wait(createDataTask);
	ghost_task_destroy(createDataTask);
#endif
	ghost_printMatrixInfo(mat);
#ifdef CHECK
	ghost_vec_t *goldLHS = ghost_createVector(context,&lvtraits);
	ghost_referenceSolver(goldLHS,matrixPath,matdt,rhs,nIter,spmvmOptions);	
#endif


	if (ghost_getRank(MPI_COMM_WORLD) == 0)
		ghost_printHeader("Performance");


	for (mode=0; mode < nModes; mode++){

		int argOptions = spmvmOptions | modes[mode];
#ifdef TASKING
//		if (modes[mode] == GHOST_SPMVM_MODE_TASKMODE) { // having a task inside a task does not work currently in this case
//#ifdef VT
//			VT_begin("foo");
//#endif
//			time = ghost_bench_spmvm(context,lhs,mat,rhs,&argOptions,nIter);
//#ifdef VT
//			VT_end("foo");
//#endif

//		} else {

			benchArgs bargs = {.ctx = context, .mat = mat, .lhs = lhs, .rhs = rhs, .spmvmOptions = &argOptions, .nIter = nIter, .time = &time};
			ghost_task_t *benchTask = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &benchFunc, &bargs, GHOST_TASK_DEFAULT);
			//ghost_task_t *benchTask = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &benchFunc, &bargs, GHOST_TASK_NO_PIN);
			ghost_task_add(benchTask);
			ghost_task_wait(benchTask);
			ghost_task_destroy(benchTask);

	//	}
#else
		time = ghost_bench_spmvm(context,lhs,mat,rhs,&argOptions,nIter);
#endif

		if (time < 0.) {
			if (ghost_getRank(MPI_COMM_WORLD) == 0)
				ghost_printLine(ghost_modeName(modes[mode]),NULL,"SKIPPED");
			continue;
		}

#ifdef CHECK
		errcount=0;
		vecdt_t res,ref;
		for (i=0; i<mat->nrows(mat); i++){
			goldLHS->entry(goldLHS,i,0,&ref);
			lhs->entry(lhs,i,0,&res);

			mytol = EPS * mat->rowLen(mat,i);
			if (creal(cabs(ref-res)) > creal(mytol) ||
					cimag(cabs(ref-res)) > cimag(mytol)){
				printf( "PE%d: error in %s, row %"PRmatIDX": %.2e + %.2ei vs. %.2e +"
						"%.2ei [ref. vs. comp.] (tol: %.2e + %.2ei, diff: %e)\n", ghost_getRank(MPI_COMM_WORLD),ghost_modeName(modes[mode]), i, creal(ref),
						cimag(ref),
						creal(res),
						cimag(res),
						creal(mytol),cimag(mytol),creal(cabs(ref-res)));
				errcount++;
				printf("PE%d: There may be more errors...\n",ghost_getRank(MPI_COMM_WORLD));
				break;
			}
		}
		ghost_midx_t totalerrors;
#ifdef GHOST_MPI
		MPI_safecall(MPI_Allreduce(&errcount,&totalerrors,1,ghost_mpi_dt_midx,MPI_SUM,MPI_COMM_WORLD));
#else
		totalerrors = errcount;
#endif
		if (totalerrors) {
			if (ghost_getRank(MPI_COMM_WORLD) == 0)
				ghost_printLine(ghost_modeName(modes[mode]),NULL,"FAILED");
		}
		else {
			ghost_mnnz_t nnz = ghost_getMatNnz(mat);
			if (ghost_getRank(MPI_COMM_WORLD) == 0)
				ghost_printLine(ghost_modeName(modes[mode]),"GF/s","%.2f",
						2*(1.e-9*nnz)/time);
		}
#else
		ghost_mnnz_t nnz = ghost_getMatNnz(mat);
		if (ghost_getRank(MPI_COMM_WORLD) == 0)
			ghost_printLine(ghost_modeName(modes[mode]),"GF/s","%.2f",2*(1.e-9*nnz)/time);
#endif
		lhs->zero(lhs);

	}
	if (ghost_getRank(MPI_COMM_WORLD) == 0)
		ghost_printFooter();

	mat->destroy(mat);
	lhs->destroy(lhs);
	rhs->destroy(rhs);
	ghost_freeContext(context);

#ifdef CHECK
	goldLHS->destroy(goldLHS);
#endif

	ghost_finish();

	return EXIT_SUCCESS;

}
