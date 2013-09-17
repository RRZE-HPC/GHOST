#include <stdio.h>
#include <ghost.h>
#include <ghost_vec.h>
#include <ghost_util.h>
#include <ghost_taskq.h>
#include <cpuid.h>
#include <strings.h>

	GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

static void rhsVal (int i, int v, void *val) 
{
	UNUSED(i);
	UNUSED(v);

	*(vecdt_t *)val = (vecdt_t)ghost_getRank(MPI_COMM_WORLD)+1;
}

static int solutionCorrect (ghost_vec_t *comp, ghost_vec_t *gold, ghost_mat_t *mat)
{
	vecdt_t ref,res,mytol;
	ghost_vidx_t i;

	for (i=0; i<comp->traits->nrows; i++){
		gold->entry(gold,i,&ref);
		comp->entry(comp,i,&res);

		mytol = 1e-10 * mat->rowLen(mat,i);
		if (creal(cabs(ref-res)) > creal(mytol) ||
				cimag(cabs(ref-res)) > cimag(mytol)){
			return 0;
		}
	}
	return 1;

}

int main(int argc, char* argv[]) 
{
	ghost_init(argc,argv);
	int nIter = 100;
	double time;
	double zero = 0.;

	ghost_pinThreads(GHOST_PIN_PHYS,NULL);
	int spmvmOptions = GHOST_SPMVM_AXPY|GHOST_SPMVM_MODE_TASKMODE;
	int mflags, vflags;
	double weight = atoi(argv[2]);
	
	
	if ((argv[3] != NULL) && !(strcasecmp(argv[3],"CUDA"))) {
		int nThreads[] = {0,2};
		int fThread[] = {0,0};
		int smt = 2;
		ghost_thpool_init(nThreads,fThread,smt);
		mflags = GHOST_SPM_DEFAULT;
		vflags = GHOST_VEC_DEFAULT;
		ghost_CUDA_init(atoi(argv[4]));
	} else {
		int nThreads[] = {ghost_getNumberOfPhysicalCores()};
		int fThread[] = {0};
		int smt = 1;
		ghost_thpool_init(nThreads,fThread,smt);
		mflags = GHOST_SPM_HOST;
		vflags = GHOST_VEC_HOST;
	}
	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = GHOST_SPM_FORMAT_SELL, .flags = mflags,.datatype = matdt);
	ghost_vtraits_t lvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS|vflags, .datatype = vecdt);
	ghost_vtraits_t rvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS|vflags, .datatype = vecdt);

	ghost_context_t *ctx;
	ghost_vec_t *lhs, *rhs;
	ghost_mat_t *mat;

	ctx = ghost_createContext(GHOST_GET_DIM_FROM_MATRIX,GHOST_GET_DIM_FROM_MATRIX,GHOST_CONTEXT_DEFAULT,argv[1],MPI_COMM_WORLD,weight);

	mat = ghost_createMatrix(ctx,&mtraits,1);
	mat->fromFile(mat,(char *)argv[1]);

	rhs = ghost_createVector(ctx,&rvtraits);
	rhs->fromFunc(rhs,rhsVal);

	lhs = ghost_createVector(ctx,&lvtraits);
	lhs->fromScalar(lhs,&zero);

	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printContextInfo(ctx);
	ghost_printMatrixInfo(mat);

	ghost_vec_t *goldLHS = ghost_createVector(ctx,&lvtraits);
	ghost_referenceSolver(&goldLHS,argv[1],matdt,ctx,rhs,nIter,spmvmOptions);	

	time = ghost_bench_spmvm(ctx,lhs,mat,rhs,&spmvmOptions,nIter);
	if (solutionCorrect(lhs,goldLHS,mat)) {

		ghost_mnnz_t nnz = ghost_getMatNnz(mat);
		if (ghost_getRank(MPI_COMM_WORLD) == 0) {
			ghost_printHeader("Performance");
			ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",ghost_flopsPerSpmvm(matdt,vecdt)*1.e-9*nnz/time);
			ghost_printFooter();
		}
	} else {
		if (ghost_getRank(MPI_COMM_WORLD) == 0) {
			ghost_printLine("Correctness check",NULL,"Failed");
		}
	}

	lhs->destroy(lhs);
	rhs->destroy(rhs);
	mat->destroy(mat);
	ghost_freeContext(ctx);

	ghost_finish();

	return EXIT_SUCCESS;
}
