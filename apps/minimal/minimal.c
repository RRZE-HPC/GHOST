#include <stdio.h>
#include <ghost.h>
#include <ghost_util.h>

#include <omp.h>
#include <unistd.h>

GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

static void rhsVal (int i, int v, void *val) 
{
	*(vecdt_t *)val = (vecdt_t)ghost_getRank()+1;
}

int main(int argc, char* argv[]) 
{
	int nIter = 1;
	double time;
	vecdt_t zero = 0.;
	vecdt_t one = 1.;

	int spmvmOptions = GHOST_SPMVM_AXPY;
	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = GHOST_SPM_FORMAT_CRS,.datatype = matdt);
	ghost_vtraits_t lvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .datatype = vecdt);
	ghost_vtraits_t rvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS, .datatype = vecdt);

	ghost_vtraits_t dmtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .nvecs=2, .datatype=vecdt);

	ghost_matfile_header_t fileheader;
	ghost_context_t *ctx;
	ghost_vec_t *lhs, *rhs, *dm, *dm2, *dm3;
	ghost_mat_t *mat;

	ghost_init(argc,argv);
//	ghost_pinThreads(GHOST_PIN_PHYS,NULL);

	ghost_readMatFileHeader(argv[1],&fileheader);

	ctx = ghost_createContext(fileheader.nrows,fileheader.ncols,GHOST_CONTEXT_DEFAULT);
	mat = ghost_createMatrix(&mtraits,1);
	rhs = ghost_createVector(&rvtraits);
	lhs = ghost_createVector(&lvtraits);

	dm = ghost_createVector(&dmtraits);
	dm2 = ghost_createVector(&dmtraits);

	mat->fromFile(mat,ctx,argv[1]);
	
	dm->fromScalar(dm,ctx,&one);
	dm2->fromScalar(dm2,ctx,&one);
	
	lhs->fromScalar(lhs,ctx,&zero);
	rhs->fromFunc(rhs,ctx,rhsVal);
	
//	lhs->print(lhs);
	dm->print(dm);
	dm2->print(dm2);

	vecdt_t alpha = 1., beta = 0.;
	ghost_gemm(dm,dm2,&dm3,&alpha, &beta, 1);

	dm3->print(dm3);

	exit(1);
	
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

	ghost_finish();

	return EXIT_SUCCESS;
}
