#include <stdio.h>
#include <ghost.h>
#include <ghost_util.h>

GHOST_REGISTER_DT_D(vecdt)

static void rhsVal (int i, int v, void *val) 
{
	UNUSED(i);
	UNUSED(v);
	*(vecdt_t *)val = 1+I*1;//i + (vecdt_t)1.0 + I*i;
}

int main( int argc, char* argv[] ) 
{
	int nIter = 1;
	double time;
	double zero = 0.;

	int ghostOptions = GHOST_OPTION_NONE;
	int spmvmOptions = GHOST_SPMVM_AXPY;
	ghost_mtraits_t mtraits = {.format = "CRS", .flags = GHOST_SPM_DEFAULT, .datatype = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL};
	ghost_vtraits_t lvtraits = {.flags = GHOST_VEC_LHS, .datatype = vecdt, .nvecs = 1};
	ghost_vtraits_t rvtraits = {.flags = GHOST_VEC_RHS, .datatype = vecdt, .nvecs = 1};

	ghost_matfile_header_t fileheader;
	ghost_context_t *ctx;
	ghost_vec_t *lhs, *rhs;
	ghost_mat_t *mat;

	ghost_init(argc,argv,ghostOptions);

	ghost_readMatFileHeader(argv[1],&fileheader);

	ctx = ghost_createContext(fileheader.nrows,GHOST_CONTEXT_DEFAULT);
	mat = ghost_createMatrix(&mtraits,1);
	rhs = ghost_createVector(&rvtraits);
	lhs = ghost_createVector(&lvtraits);

	mat->fromBin(mat,argv[1],ctx);
	lhs->fromScalar(lhs,ctx,&zero);
	rhs->fromFunc(rhs,ctx,rhsVal);

	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printOptionsInfo(ghostOptions);
	ghost_printContextInfo(ctx);

	ghost_printHeader("Performance");
	time = ghost_bench_spmvm(ctx,lhs,mat,rhs,&spmvmOptions,nIter);

	if (time > 0.)
		ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",FLOPS_PER_ENTRY*1.e-9*mat->nnz(mat)/time);

	ghost_printFooter();

	lhs->destroy(lhs);
	rhs->destroy(rhs);
	ghost_freeContext(ctx);

	ghost_finish();

	return EXIT_SUCCESS;
}
