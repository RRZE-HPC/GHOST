#include <stdio.h>

#include <ghost.h>
#include <ghost_util.h>
#include <ghost_vec.h>

static ghost_vdat_t rhsVal (int i) 
{
	return i + (ghost_vdat_t)1.0;
}

int main( int argc, char* argv[] ) 
{
	int nIter = 100;
	double time;
	int mode = GHOST_MODE_NOMPI;
	int options = GHOST_OPTION_AXPY;
	ghost_mtraits_t trait = {.format = "CRS", .flags = GHOST_SPM_DEFAULT, .aux = NULL};

	ghost_context_t *ctx;
	ghost_vec_t *lhs;
	ghost_vec_t *rhs;

	ghost_init(argc,argv,options);
	ctx = ghost_createContext(argv[1],&trait,1,GHOST_CONTEXT_DEFAULT);
	rhs = ghost_createVector(ctx,GHOST_VEC_RHS,rhsVal); // RHS vec
	lhs = ghost_createVector(ctx,GHOST_VEC_LHS,NULL);   // LHS vec (=0)

	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printOptionsInfo(options);
	ghost_printContextInfo(ctx);
	
	ghost_printHeader("Performance");
	
	time = ghost_bench_spmvm(lhs,ctx,rhs,mode,options,nIter);
	if (time > 0)
		ghost_printLine(ghost_modeName(mode),"GF/s","%.2f",FLOPS_PER_ENTRY*1.e-9*ctx->gnnz(ctx)/time);

	ghost_printFooter();

	ghost_freeVector(lhs);
	ghost_freeVector(rhs);
	ghost_freeContext(ctx);

	ghost_finish();

	return EXIT_SUCCESS;
}
