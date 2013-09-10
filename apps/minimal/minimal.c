#include <stdio.h>
#include <ghost.h>
#include <ghost_vec.h>
#include <ghost_util.h>
#include <ghost_taskq.h>
#include <cpuid.h>


GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

static void rhsVal (int i, int v, void *val) 
{
	UNUSED(i);
	UNUSED(v);

	*(vecdt_t *)val = (vecdt_t)ghost_getRank(MPI_COMM_WORLD)+1;
}

static void *minimalTask(void *arg)
{
	int nIter = 100;
	double time;
	double zero = 0.;

	int spmvmOptions = GHOST_SPMVM_AXPY|GHOST_SPMVM_MODE_GOODFAITH;
	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = GHOST_SPM_FORMAT_CRS,.datatype = matdt);
	ghost_vtraits_t lvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .datatype = vecdt);
	ghost_vtraits_t rvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS, .datatype = vecdt);

	ghost_context_t *ctx;
	ghost_vec_t *lhs, *rhs;
	ghost_mat_t *mat;

	ctx = ghost_createContext(GHOST_GET_DIM_FROM_MATRIX,GHOST_GET_DIM_FROM_MATRIX,GHOST_CONTEXT_DEFAULT,arg,MPI_COMM_WORLD);
	mat = ghost_createMatrix(ctx,&mtraits,1);
	rhs = ghost_createVector(ctx,&rvtraits);
	lhs = ghost_createVector(ctx,&lvtraits);

	mat->fromFile(mat,(char *)arg);
	lhs->fromScalar(lhs,&zero);
	rhs->fromFunc(rhs,rhsVal);
	
	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printContextInfo(ctx);
	ghost_printMatrixInfo(mat);

	time = ghost_bench_spmvm(ctx,lhs,mat,rhs,&spmvmOptions,nIter);
	
	ghost_mnnz_t nnz = ghost_getMatNnz(mat);
	if (ghost_getRank(MPI_COMM_WORLD) == 0) {
		ghost_printHeader("Performance");
		ghost_printLine(ghost_modeName(spmvmOptions),"GF/s","%.2f",ghost_flopsPerSpmvm(matdt,vecdt)*1.e-9*nnz/time);
		ghost_printFooter();
	}
	
	lhs->destroy(lhs);
	rhs->destroy(rhs);
	mat->destroy(mat);
	ghost_freeContext(ctx);

	return NULL;
}

int main(int argc, char* argv[]) 
{
	ghost_init(argc,argv);

//#ifdef MIC 
	// SMT-3
//	ghost_thpool_init(ghost_cpuid_topology.numHWThreads/4*3);
//#else 
	// no SMT
//	ghost_thpool_init(ghost_getNumberOfPhysicalCores());
//#endif
//	ghost_taskq_init(ghost_cpuid_topology.numSockets);

	ghost_task_t *t = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &minimalTask, argv[1], GHOST_TASK_DEFAULT);
	ghost_task_add(t);

	ghost_task_wait(t);
	ghost_task_destroy(t);
	ghost_finish();

	return EXIT_SUCCESS;
}
