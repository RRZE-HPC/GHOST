#include <stdio.h>
#include <ghost.h>
#include <ghost_util.h>
#include <ghost_vec.h>

	GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

static void rhsVal (int i, int v, void *val) 
{
	UNUSED(i);
	UNUSED(v);

	*(vecdt_t *)val = (vecdt_t)ghost_getRank(MPI_COMM_WORLD)+1;
}

int main(int argc, char* argv[]) 
{
	int nIter = 100;
	double time;
	double zero = 0.;
	int nranks;
	int *ranks0, *ranks1;
	int i;
	int spmvmOptions = GHOST_SPMVM_AXPY;

	ghost_init(argc,argv);
	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_pinThreads(GHOST_PIN_PHYS,NULL);

	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = GHOST_SPM_FORMAT_CRS,.datatype = matdt);
	ghost_vtraits_t lvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .datatype = vecdt);
	ghost_vtraits_t rvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS, .datatype = vecdt);

	ghost_context_t *ctx;
	ghost_vec_t *lhs, *rhs;
	ghost_mat_t *mat;

	MPI_Group group, group0, group1;
	MPI_Comm comm, comm0, comm1;

	MPI_safecall(MPI_Comm_group(MPI_COMM_WORLD,&group));

	nranks = ghost_getNumberOfRanks(MPI_COMM_WORLD);

	ranks0 = ghost_malloc(nranks/2*sizeof(int));
	ranks1 = ghost_malloc((nranks/2+nranks%2)*sizeof(int));

	for (i=0; i<nranks; i++) {
		if (i<nranks/2) {
			ranks0[i] = i;
		} else {
			ranks1[i-nranks/2] = i;
		}
	}

	MPI_safecall(MPI_Group_incl(group,nranks/2,ranks0,&group0));
	MPI_safecall(MPI_Group_incl(group,nranks/2+nranks%2,ranks1,&group1));

	MPI_safecall(MPI_Comm_create(MPI_COMM_WORLD,group0,&comm0));
	MPI_safecall(MPI_Comm_create(MPI_COMM_WORLD,group1,&comm1));

	if (ghost_getRank(MPI_COMM_WORLD) < nranks/2) {
		comm = comm0;
	} else {
		comm = comm1;
	}

	if (ghost_getRank(MPI_COMM_WORLD) < nranks/2) {
		ctx = ghost_createContext(GHOST_GET_DIM_FROM_MATRIX,GHOST_GET_DIM_FROM_MATRIX,GHOST_CONTEXT_DEFAULT,argv[1],comm);
		mat = ghost_createMatrix(ctx,&mtraits,1);
		rhs = ghost_createVector(ctx,&rvtraits);
		lhs = ghost_createVector(ctx,&lvtraits);

		mat->fromFile(mat,argv[1]);
		lhs->fromScalar(lhs,&zero);
		rhs->fromFunc(rhs,rhsVal);

		ghost_printContextInfo(ctx);
		ghost_printMatrixInfo(mat);
		time = ghost_bench_spmvm(ctx,lhs,mat,rhs,&spmvmOptions,nIter);

		double gflops = ghost_flopsPerSpmvm(matdt,vecdt)*1.e-9*ghost_getMatNnz(mat)/time;
		if (ghost_getRank(comm) == 0) {
			ghost_printHeader("Performance");
			ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",gflops);
			ghost_printFooter();
		}
		lhs->destroy(lhs);
		rhs->destroy(rhs);
		mat->destroy(mat);
		ghost_freeContext(ctx);
	} else {
		ctx = ghost_createContext(GHOST_GET_DIM_FROM_MATRIX,GHOST_GET_DIM_FROM_MATRIX,GHOST_CONTEXT_DEFAULT,argv[2],comm);
		mat = ghost_createMatrix(ctx,&mtraits,1);
		rhs = ghost_createVector(ctx,&rvtraits);
		lhs = ghost_createVector(ctx,&lvtraits);

		mat->fromFile(mat,argv[2]);
		lhs->fromScalar(lhs,&zero);
		rhs->fromFunc(rhs,rhsVal);

		ghost_printContextInfo(ctx);
		ghost_printMatrixInfo(mat);
		time = ghost_bench_spmvm(ctx,lhs,mat,rhs,&spmvmOptions,nIter);

		double gflops = ghost_flopsPerSpmvm(matdt,vecdt)*1.e-9*ghost_getMatNnz(mat)/time;
		if (ghost_getRank(comm) == 0) {
			ghost_printHeader("Performance");
			ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",gflops);
			ghost_printFooter();
		}
		lhs->destroy(lhs);
		rhs->destroy(rhs);
		mat->destroy(mat);
		ghost_freeContext(ctx);
	}




	ghost_finish();

	return EXIT_SUCCESS;
}
