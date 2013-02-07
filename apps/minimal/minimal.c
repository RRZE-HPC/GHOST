#include <stdio.h>
#include <ghost.h>
#include <ghost_util.h>

GHOST_REGISTER_DT_D(vecdt)

static void rhsVal (int i, void *val) 
{
	*(vecdt_t *)val = i + (vecdt_t)1.0;
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
		.datatype = GHOST_BINCRS_DT_FLOAT};
	ghost_vtraits_t lvtraits = {.flags = GHOST_VEC_LHS,.aux = NULL,.datatype = vecdt};
	ghost_vtraits_t rvtraits = {.flags = GHOST_VEC_RHS,.aux = NULL,.datatype = vecdt};

	ghost_context_t *ctx;
	ghost_vec_t *lhs;
	ghost_vec_t *rhs;

	ghost_init(argc,argv,ghostOptions);
	ctx = ghost_createContext(argv[1],&mtraits,1,GHOST_CONTEXT_DEFAULT);
	rhs = ghost_createVector(ctx,&rvtraits); // RHS vec
	lhs = ghost_createVector(ctx,&lvtraits);   // LHS vec (=0)

	rhs->fromFunc(rhs,rhsVal);

	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printOptionsInfo(ghostOptions);
	ghost_printContextInfo(ctx);

	ghost_printHeader("Performance");
	
	time = ghost_bench_spmvm(lhs,ctx,rhs,&spmvmOptions,nIter);

	ghost_normalizeVec(lhs);
	
	/*vecdt n;
	ghost_dotProduct(lhs,lhs,&n);
	lhs->toFile(lhs,"/tmp/lhs.dump",0,0);
	lhs->fromFile(lhs,"/tmp/lhs.dump",0);

	lhs->print(lhs);
	ghost_vecToFile(lhs,"/tmp/lhs_global.dump",ctx);
	ghost_vecFromFile(lhs,"/tmp/lhs_global.dump",ctx);

	lhs->print(lhs);*/


	if (time > 0)
		ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",FLOPS_PER_ENTRY*1.e-9*ctx->gnnz(ctx)/time);

	ghost_printFooter();


	lhs->destroy(lhs);
	rhs->destroy(rhs);
	ghost_freeContext(ctx);

	ghost_finish();

	return EXIT_SUCCESS;
}
