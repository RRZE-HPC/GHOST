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

static void *mywork(void *nt)
{
	sleep(2);
	printf("in mywork after sleeping with arg %d\n",*(int *)nt);

#pragma omp parallel 
	printf("in mywork: openmp thread no. %d on core %d\n",omp_get_thread_num(),ghost_getCore());

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

	int arg = 42;
	ghost_task_t mytask = ghost_spawnTask(&mywork,&arg,3,NULL,"mytask",GHOST_TASK_ASYNC);

	ghost_readMatFileHeader(argv[1],&fileheader);

	ctx = ghost_createContext(fileheader.nrows,GHOST_CONTEXT_DEFAULT);
	mat = ghost_createMatrix(&mtraits,1);
	rhs = ghost_createVector(&rvtraits);
	lhs = ghost_createVector(&lvtraits);

	mat->fromFile(mat,ctx,argv[1]);
	lhs->fromScalar(lhs,ctx,&zero);
	rhs->fromFunc(rhs,ctx,rhsVal);
	
	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printContextInfo(ctx);
	ghost_printMatrixInfo(mat);

	ghost_printHeader("Performance");
	time = ghost_bench_spmvm(ctx,lhs,mat,rhs,&spmvmOptions,nIter);


	if (time > 0.)
		ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",FLOPS_PER_ENTRY*1.e-9*ghost_getMatNnz(mat)/time);


	ghost_printFooter();

	lhs->destroy(lhs);
	rhs->destroy(rhs);
	ghost_freeContext(ctx);

	ghost_finish();
	ghost_waitTask(&mytask);

	return EXIT_SUCCESS;
}
