#include <stdio.h>
#include <ghost.h>
#include <ghost_vec.h>
#include <ghost_util.h>
#include <ghost_taskq.h>

	
GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

int main(int argc, char ** argv)
{
	vecdt_t one = 1.;
	vecdt_t two = 2.;
	vecdt_t scale = 2.;
	vecdt_t shift = -0.;
	int spmvmOptions = GHOST_SPMVM_AXPY|GHOST_SPMVM_APPLY_SCALE|GHOST_SPMVM_APPLY_SHIFT;
	
	ghost_init(argc,argv);

	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = GHOST_SPM_FORMAT_CRS,.datatype = matdt,.scale=&scale,.shift=&shift);
	ghost_vtraits_t lvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .datatype = vecdt);
	ghost_vtraits_t rvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS, .datatype = vecdt);

	ghost_context_t *ctx;
	ghost_vec_t *lhs, *rhs;
	ghost_mat_t *mat;

	ctx = ghost_createContext(GHOST_GET_DIM_FROM_MATRIX,GHOST_GET_DIM_FROM_MATRIX,GHOST_CONTEXT_DEFAULT,argv[1],MPI_COMM_WORLD,1.);
	mat = ghost_createMatrix(ctx,&mtraits,1);
	mat->fromFile(mat,(char *)argv[1]);
	rhs = ghost_createVector(ctx,&rvtraits);
	lhs = ghost_createVector(ctx,&lvtraits);

	lhs->fromScalar(lhs,&one);
	rhs->fromScalar(rhs,&two);
	
	ghost_spmvm(ctx,lhs,mat,rhs,&spmvmOptions);

	lhs->print(lhs);
	
	lhs->destroy(lhs);
	rhs->destroy(rhs);
	mat->destroy(mat);
	ghost_freeContext(ctx);
	
	ghost_finish();
}
