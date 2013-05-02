#include <stdio.h>
#include <ghost.h>
#include <ghost_util.h>

#include <omp.h>
#include <unistd.h>

GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

static void rhsVal (int i, int v, void *val) 
{
	UNUSED(i);
	UNUSED(v);
	*(vecdt_t *)val = 1+I*1;//i + (vecdt_t)1.0 + I*i;
}

/*static void *mywork(void *t)
{
	sleep(*(int *)t);
	printf("in mywork after sleeping for %d seconds\n",*(int *)t);

#pragma omp parallel 
	printf("in mywork: openmp thread no. %d on core %d\n",omp_get_thread_num(),ghost_getCore());

	return NULL;
}*/


typedef struct {
	ghost_context_t *ctx;
	ghost_mat_t *mat;
	ghost_vec_t *lhs, *rhs;
	int *spmvmOptions;
	int nIter;
	double *time;
} benchArgs;

typedef struct {
	ghost_context_t *ctx;
	ghost_mat_t *mat;
	ghost_vec_t *lhs, *rhs;
	char *matfile;
	vecdt_t *lhsInit;
	void (*rhsInit)(int,int,void*);
} createDataArgs;


static void *createDataTask(void *vargs)
{
	createDataArgs *args = (createDataArgs *)vargs;
	args->mat->fromFile(args->mat,args->ctx,args->matfile);
	args->lhs->fromScalar(args->lhs,args->ctx,args->lhsInit);
	args->rhs->fromFunc(args->rhs,args->ctx,args->rhsInit);

	return NULL;
}

static void *benchTask(void *vargs)
{
	benchArgs *args = (benchArgs *)vargs;
	*(args->time) = ghost_bench_spmvm(args->ctx,args->lhs,args->mat,args->rhs,args->spmvmOptions,args->nIter);

	return NULL;
}


int main( int argc, char* argv[] ) 
{
	int nIter = 100;
	double time;
	vecdt_t zero = 0.;

	int spmvmOptions = GHOST_SPMVM_AXPY;
	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.datatype = matdt);
	ghost_vtraits_t lvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .datatype = vecdt);
	ghost_vtraits_t rvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS, .datatype = vecdt);

	ghost_matfile_header_t fileheader;
	ghost_context_t *ctx;
	ghost_vec_t *lhs, *rhs;
	ghost_mat_t *mat;


	ghost_init(argc,argv);
	ghost_pinThreads(GHOST_PIN_PHYS,NULL);

	ghost_readMatFileHeader(argv[1],&fileheader);

	ctx = ghost_createContext(fileheader.nrows,GHOST_CONTEXT_DEFAULT);
	mat = ghost_createMatrix(&mtraits,1);
	rhs = ghost_createVector(&rvtraits);
	lhs = ghost_createVector(&lvtraits);

	/*mat->fromFile(mat,ctx,argv[1]);
	lhs->fromScalar(lhs,ctx,&zero);
	rhs->fromFunc(rhs,ctx,rhsVal);*/
	createDataArgs args = {.ctx = ctx, .mat = mat, .lhs = lhs, .rhs = rhs, .matfile = argv[1], .lhsInit = &zero, .rhsInit = rhsVal};
	ghost_task_t cdTask = ghost_spawnTask(&createDataTask,&args,6,NULL,"create data structures",GHOST_TASK_ASYNC);
	
	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printContextInfo(ctx);

	ghost_waitTask(&cdTask);
	ghost_printMatrixInfo(mat);

	ghost_printHeader("Performance");

	benchArgs bargs = {.ctx = ctx, .mat = mat, .lhs = lhs, .rhs = rhs, .spmvmOptions = &spmvmOptions, .nIter = nIter, .time = &time};
	ghost_spawnTask(&benchTask,&bargs,GHOST_TASK_ALIKE,&cdTask,"bench",GHOST_TASK_SYNC);
//	time = ghost_bench_spmvm(ctx,lhs,mat,rhs,&spmvmOptions,nIter);


	if (time > 0.)
		ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",FLOPS_PER_ENTRY*1.e-9*ghost_getMatNnz(mat)/time);


	ghost_printFooter();

	
	lhs->destroy(lhs);
	rhs->destroy(rhs);
	ghost_freeContext(ctx);

	ghost_finish();

	return EXIT_SUCCESS;
}
