#include <stdio.h>
#include <ghost.h>
#include <ghost_util.h>
#include <ghost_taskq.h>


GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

static void rhsVal (int i, int v, void *val) 
{
	UNUSED(i);
	UNUSED(v);

	*(vecdt_t *)val = (vecdt_t)ghost_getRank()+1;
}

static void *minimalTask(void *arg)
{
	UNUSED(arg);
	int nIter = 1;
	double time;
	double zero = 0.;

	int spmvmOptions = GHOST_SPMVM_AXPY;
	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = GHOST_SPM_FORMAT_CRS,.datatype = matdt);
	ghost_vtraits_t lvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .datatype = vecdt);
	ghost_vtraits_t rvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS, .datatype = vecdt);

	ghost_matfile_header_t fileheader;
	ghost_context_t *ctx;
	ghost_vec_t *lhs, *rhs;
	ghost_mat_t *mat;
	//ghost_pinThreads(GHOST_PIN_PHYS,NULL);

	ghost_readMatFileHeader((char *)arg,&fileheader);

	ctx = ghost_createContext(fileheader.nrows,fileheader.ncols,GHOST_CONTEXT_DEFAULT,arg);
	mat = ghost_createMatrix(&mtraits,1);
	rhs = ghost_createVector(&rvtraits);
	lhs = ghost_createVector(&lvtraits);

	mat->fromFile(mat,ctx,(char *)arg);
	lhs->fromScalar(lhs,ctx,&zero);
	rhs->fromFunc(rhs,ctx,rhsVal);
	
	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printContextInfo(ctx);
	ghost_printMatrixInfo(mat);

	ghost_printHeader("Performance");
	time = ghost_bench_spmvm(ctx,lhs,mat,rhs,&spmvmOptions,nIter);
	ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",ghost_flopsPerSpmvm(matdt,vecdt)*1.e-9*ghost_getMatNnz(mat)/time);
	ghost_printFooter();
	
	lhs->destroy(lhs);
	rhs->destroy(rhs);
	mat->destroy(mat);
	ghost_freeContext(ctx);

	return NULL;

}

int main(int argc, char* argv[]) 
{

	ghost_init(argc,argv);
	ghost_task_t *t = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &minimalTask, argv[1], GHOST_TASK_DEFAULT);
	ghost_task_add(t);

	ghost_finish();

	return EXIT_SUCCESS;
}
