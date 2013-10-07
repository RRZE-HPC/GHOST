#include <stdio.h>
#include <ghost.h>
#include <ghost_vec.h>
#include <ghost_util.h>
#include <ghost_taskq.h>
#include <omp.h>


GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)
	


static void rhsVal (int i, int v, void *val) 
{
	UNUSED(i);
	UNUSED(v);

	*(vecdt_t *)val = (vecdt_t)ghost_getRank(MPI_COMM_WORLD)+1;
}


static void *task(void *arg)
{
	ghost_context_t *ctx;
	ghost_vec_t *lhs, *rhs;
	ghost_mat_t *mat;
	double zero = 0.;
	double time;
	int spmvmOptions = GHOST_SPMVM_AXPY|GHOST_SPMVM_MODE_TASKMODE;
	//int spmvmOptions = GHOST_SPMVM_AXPY|GHOST_SPMVM_MODE_GOODFAITH;
//	int spmvmOptions = GHOST_SPMVM_AXPY|GHOST_SPMVM_MODE_NOMPI;
	int nIter = 100;

	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = GHOST_SPM_FORMAT_CRS,.datatype = matdt);
	ghost_vtraits_t lvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .datatype = vecdt);
	ghost_vtraits_t rvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS, .datatype = vecdt);


	ctx =
		ghost_createContext(GHOST_GET_DIM_FROM_MATRIX,GHOST_GET_DIM_FROM_MATRIX,GHOST_CONTEXT_DEFAULT,arg,MPI_COMM_WORLD,1);
	mat = ghost_createMatrix(ctx,&mtraits,1);
	mat->fromFile(mat,(char *)arg);
	rhs = ghost_createVector(ctx,&rvtraits);
	lhs = ghost_createVector(ctx,&lvtraits);

	lhs->fromScalar(lhs,&zero);
	rhs->fromFunc(rhs,rhsVal);
	UNUSED(arg);

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
	int nt[] = {12,12};
	int ft[] = {0,0};
	int levels = 1;

	ghost_init(argc,argv);
	ghost_tasking_init(nt,ft,levels);
	
	ghost_pinThreads(GHOST_PIN_PHYS,NULL);
	ghost_task_t *t = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &task, argv[1], GHOST_TASK_DEFAULT);
	ghost_task_add(t);
	ghost_task_wait(t);
	ghost_task_destroy(t);

	ghost_tasking_finish(nt,ft,levels);
	ghost_finish();

	return EXIT_SUCCESS;
}

