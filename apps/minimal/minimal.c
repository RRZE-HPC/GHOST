#include <stdio.h>
#include <ghost.h>
#include <ghost_util.h>

#define MPI

#ifdef MPI
#include <mpi.h>
#define MPI_VECDT MPI_DOUBLE
#endif


typedef double vecdt;
#define VECDT GHOST_BINCRS_DT_DOUBLE

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
	ghost_vtraits_t lvtraits = {.flags = GHOST_VEC_LHS,.aux = NULL,.datatype = VECDT};
	ghost_vtraits_t rvtraits = {.flags = GHOST_VEC_RHS,.aux = NULL,.datatype = VECDT};

	ghost_context_t *ctx;
	ghost_vec_t *lhs;
	ghost_vec_t *rhs;

	ghost_init(argc,argv,ghostOptions);
	ctx = ghost_createContext(argv[1],&mtraits,1,GHOST_CONTEXT_DEFAULT);
	rhs = ghost_createVector(ctx,&rvtraits); // RHS vec
	lhs = ghost_createVector(ctx,&lvtraits);   // LHS vec (=0)

	rhs->fromFP(rhs,ctx->communicator,rhsVal);	

	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printOptionsInfo(ghostOptions);
	ghost_printContextInfo(ctx);

	ghost_printHeader("Performance");
	
	time = ghost_bench_spmvm(lhs,ctx,rhs,&spmvmOptions,nIter);

	vecdt n;
	lhs->dotProduct(lhs,lhs,&n);
#ifdef MPI
	MPI_safecall(MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_VECDT, MPI_SUM, MPI_COMM_WORLD));
#endif
	n = 1/sqrt(n);
	lhs->scale(lhs,&n);
	
	lhs->dotProduct(lhs,lhs,&n);
#ifdef MPI
	MPI_safecall(MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_VECDT, MPI_SUM, MPI_COMM_WORLD));
#endif
	printf("%f should be 1.0\n",n);


	if (time > 0)
		ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",FLOPS_PER_ENTRY*1.e-9*ctx->gnnz(ctx)/time);

	ghost_printFooter();


	lhs->destroy(lhs);
	rhs->destroy(rhs);
	ghost_freeContext(ctx);

	ghost_finish();

	return EXIT_SUCCESS;
}
