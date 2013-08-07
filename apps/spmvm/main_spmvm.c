#include <ghost.h>
#include <ghost_util.h>
#include <ghost_taskq.h>

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

//#define TASKING
//#define CHECK // compare with reference solution

GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

#ifdef TASKING
typedef struct {
	ghost_context_t *ctx;
	ghost_mat_t *mat;
	ghost_vec_t *lhs, *rhs;
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
	args->mat->fromFile(args->mat,args->ctx,args->matfile);
	args->lhs->fromScalar(args->lhs,args->ctx,args->lhsInit);
	args->rhs->fromFunc(args->rhs,args->ctx,args->rhsInit);

	return NULL;
}
static void *benchFunc(void *vargs)
{
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

	int  mode, nIter = 5;
	double time;
	vecdt_t zero = 0.;
	matdt_t shift = 0.;

#ifdef CHECK
	ghost_midx_t i, errcount = 0;
	double mytol;
#endif

	int modes[] = {GHOST_SPMVM_MODE_NOMPI,
		GHOST_SPMVM_MODE_VECTORMODE/*,
		GHOST_SPMVM_MODE_GOODFAITH,
		GHOST_SPMVM_MODE_TASKMODE*/};
	int nModes = sizeof(modes)/sizeof(int);

	int spmvmOptions = GHOST_SPMVM_AXPY /* | GHOST_SPMVM_APPLY_SHIFT*/;
	ghost_matfile_header_t fileheader;

	ghost_mat_t *mat; // matrix
	ghost_vec_t *lhs; // lhs vector
	ghost_vec_t *rhs; // rhs vector

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

	ghost_readMatFileHeader(matrixPath,&fileheader);
	context = ghost_createContext(fileheader.nrows,fileheader.ncols,GHOST_CONTEXT_DEFAULT);
	mat = ghost_createMatrix(&mtraits,1);
	lhs = ghost_createVector(&lvtraits);
	rhs = ghost_createVector(&rvtraits);

#ifdef TASKING
	createDataArgs args = {.ctx = context, .mat = mat, .lhs = lhs, .rhs = rhs, .matfile = matrixPath, .lhsInit = &zero, .rhsInit = rhsVal};
	ghost_task_t *createDataTask = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &createDataFunc, &args, GHOST_TASK_DEFAULT);
	ghost_task_add(createDataTask);
#else
	mat->fromFile(mat,context,matrixPath);
	lhs->fromScalar(lhs,context,&zero);
	rhs->fromFunc(rhs,context,&rhsVal);
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
	ghost_vec_t *goldLHS = ghost_referenceSolver(matrixPath,matdt,context,rhs,nIter,spmvmOptions);	
#endif

	ghost_printHeader("Performance");

	for (mode=0; mode < nModes; mode++){

		int argOptions = spmvmOptions | modes[mode];
#ifdef TASKING
		if (modes[mode] == GHOST_SPMVM_MODE_TASKMODE) { // having a task inside a task does not work currently in this case
			time = ghost_bench_spmvm(context,lhs,mat,rhs,&argOptions,nIter);

		} else {

			benchArgs bargs = {.ctx = context, .mat = mat, .lhs = lhs, .rhs = rhs, .spmvmOptions = &argOptions, .nIter = nIter, .time = &time};
			ghost_task_t *benchTask = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &benchFunc, &bargs, GHOST_TASK_DEFAULT);
			ghost_task_add(benchTask);
			ghost_task_wait(benchTask);
			ghost_task_destroy(benchTask);

		}
#else
		time = ghost_bench_spmvm(context,lhs,mat,rhs,&argOptions,nIter);
#endif


		if (time < 0.) {
			ghost_printLine(ghost_modeName(modes[mode]),NULL,"SKIPPED");
			continue;
		}

#ifdef CHECK
		errcount=0;
		vecdt_t res,ref;
		for (i=0; i<mat->nrows(mat); i++){
			goldLHS->entry(goldLHS,i,&ref);
			lhs->entry(lhs,i,&res);

			mytol = 1e-7 * mat->rowLen(mat,i);
			if (creal(cabs(ref-res)) > creal(mytol) ||
					cimag(cabs(ref-res)) > cimag(mytol)){
				printf( "PE%d: error in %s, row %"PRmatIDX": %.2e + %.2ei vs. %.2e +"
						"%.2ei (tol: %.2e + %.2ei, diff: %e)\n", ghost_getRank(),ghost_modeName(modes[mode]), i, creal(ref),
						cimag(ref),
						creal(res),
						cimag(res),
						creal(mytol),cimag(mytol),creal(cabs(ref-res)));
				errcount++;
				//break;
			}
		}
		ghost_midx_t totalerrors;
#ifdef GHOST_MPI
		MPI_safecall(MPI_Allreduce(&errcount,&totalerrors,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD));
#else
		totalerrors = errcount;
#endif
		if (totalerrors)
			ghost_printLine(ghost_modeName(modes[mode]),NULL,"FAILED");
		else
			ghost_printLine(ghost_modeName(modes[mode]),"GF/s","%.2f",
					2*(1.e-9*
					ghost_getMatNnz(mat))/time);
#else
		ghost_printLine(ghost_modeName(modes[mode]),"GF/s","%.2f",2*(1.e-9*ghost_getMatNnz(mat))/time);
#endif

		lhs->zero(lhs);

	}
	ghost_printFooter();

	lhs->destroy(lhs);
	rhs->destroy(rhs);
	ghost_freeContext( context );

#ifdef CHECK
	goldLHS->destroy(goldLHS);
#endif

	ghost_finish();

	return EXIT_SUCCESS;

}
