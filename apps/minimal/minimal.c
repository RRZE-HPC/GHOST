#include <stdio.h>

#include <ghost.h>
#include <ghost_util.h>

typedef double vecdt;

static void rhsVal (int i, void *val) 
{
	*(double *)val = i + (vecdt)1.0;
}

int main( int argc, char* argv[] ) 
{
	int nIter = 1;
	double time;
	int ghostOptions = GHOST_OPTION_NONE;
	int spmvmOptions = GHOST_SPMVM_AXPY;
	ghost_mtraits_t mtraits = {.format = "CRS", 
		.flags = GHOST_SPM_DEFAULT, 
		.aux = NULL, 
		.datatype = GHOST_BINCRS_DT_DOUBLE};

	ghost_context_t *ctx;
	ghost_vec_t *lhs;
	ghost_vec_t *rhs;

	ghost_init(argc,argv,ghostOptions);
	ctx = ghost_createContext(argv[1],&mtraits,1,GHOST_CONTEXT_DEFAULT);
	rhs = ghost_createVector(ctx,GHOST_VEC_RHS,rhsVal); // RHS vec
	lhs = ghost_createVector(ctx,GHOST_VEC_LHS,NULL);   // LHS vec (=0)

	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printOptionsInfo(ghostOptions);
	ghost_printContextInfo(ctx);

	ghost_printHeader("Performance");
	
	time = ghost_bench_spmvm(lhs,ctx,rhs,&spmvmOptions,nIter);
	int i;
	for (i=0; i<ctx->lnrows(ctx); i++)
		printf("%d: %d: %f\n",ghost_getRank(), i, ((double *)(lhs->val))[i]);
	if (time > 0)
		ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",FLOPS_PER_ENTRY*1.e-9*ctx->gnnz(ctx)/time);

	ghost_printFooter();


//	lhs->destroy(lhs);
//	rhs->destroy(rhs);
	ghost_freeContext(ctx);

	ghost_finish();

	return EXIT_SUCCESS;
}
