#include <ghost_config.h>
#include <ghost_types.h>
#include <ghost_constants.h>
#include <ghost_util.h>
#include <ghost_math.h>
#include <ghost_affinity.h>
#include <ghost_blas_mangle.h>
#include <strings.h>
#include <math.h>
#include <complex.h>

void ghost_dotProduct(ghost_vec_t *vec, ghost_vec_t *vec2, void *res)
{
	vec->dotProduct(vec,vec2,res);
#ifdef GHOST_HAVE_MPI
	int v;
	if (!(vec->traits->flags & GHOST_VEC_GLOBAL)) {
		for (v=0; v<MIN(vec->traits->nvecs,vec2->traits->nvecs); v++) {
			MPI_safecall(MPI_Allreduce(MPI_IN_PLACE, (char *)res+ghost_sizeofDataType(vec->traits->datatype)*v, 1, ghost_mpi_dataType(vec->traits->datatype), ghost_mpi_op_sum(vec->traits->datatype), vec->context->mpicomm));
		}
	}
#endif

}

void ghost_normalizeVec(ghost_vec_t *vec)
{
	if (vec->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
		if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
			complex float res;
			ghost_dotProduct(vec,vec,&res);
			res = 1.f/csqrtf(res);
			vec->scale(vec,&res);
		} else {
			float res;
			ghost_dotProduct(vec,vec,&res);
			res = 1.f/sqrtf(res);
			vec->scale(vec,&res);
		}
	} else {
		if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
			complex double res;
			ghost_dotProduct(vec,vec,&res);
			res = 1./csqrt(res);
			vec->scale(vec,&res);
		} else {
			double res;
			ghost_dotProduct(vec,vec,&res);
			res = 1./sqrt(res);
			vec->scale(vec,&res);
		}
	}
}

int ghost_spmvm(ghost_context_t *context, ghost_vec_t *res, ghost_mat_t *mat, ghost_vec_t *invec, 
		int *spmvmOptions)
{
	ghost_solver_t solver = NULL;
	ghost_pickSpMVMMode(context,spmvmOptions);
	solver = context->solvers[ghost_getSpmvmModeIdx(*spmvmOptions)];

	if (!solver)
		return GHOST_FAILURE;

	solver(context,res,mat,invec,*spmvmOptions);

	return GHOST_SUCCESS;
}

int ghost_gemm(char *transpose, ghost_vec_t *v, ghost_vec_t *w, ghost_vec_t *x, void *alpha, void *beta, int reduce)
{
	if ((v->traits->flags & GHOST_VEC_SCATTERED) || 
			(w->traits->flags & GHOST_VEC_SCATTERED) ||
			(x->traits->flags & GHOST_VEC_SCATTERED)) {
		WARNING_LOG("Scattered vectors currently not supported in ghost_gemm()");
		return GHOST_FAILURE;
	}
	ghost_midx_t nrV,ncV,nrW,ncW,nrX,ncX;
	// TODO if rhs vector data will not be continous
	complex double zero = 0.+I*0.;
	if ((!strcmp(transpose,"N"))||(!strcmp(transpose,"n")))
	{
		nrV=v->traits->nrows; ncV=v->traits->nvecs;
	}
	else
	{
		nrV=v->traits->nvecs; ncV=v->traits->nrows;
	}
	nrW=w->traits->nrows; ncW=w->traits->nvecs;
	nrX=x->traits->nrows; ncX=w->traits->nvecs;
	if (ncV!=nrW || nrV!=nrX || ncW!=ncX) {
		WARNING_LOG("GEMM with incompatible vectors!");
		return GHOST_FAILURE;
	}
	if (v->traits->datatype != w->traits->datatype) {
		WARNING_LOG("GEMM with vector of different datatype does not work");
		return GHOST_FAILURE;
	}

#ifdef LONGIDX // TODO
	ABORT("GEMM with LONGIDX not implemented");
#endif

	ghost_vidx_t m,n,k;
	m = nrV;
	k = ncV;
	n = ncW;

	if (v->traits->datatype != w->traits->datatype) {
		ABORT("Dgemm with mixed datatypes does not work!");
	}

	DEBUG_LOG(1,"Calling XGEMM with (%"PRvecIDX"x%"PRvecIDX") * (%"PRvecIDX"x%"PRvecIDX") = (%"PRvecIDX"x%"PRvecIDX")",m,k,k,n,m,n);

	if (v->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
		if (v->traits->datatype & GHOST_BINCRS_DT_DOUBLE) {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(v->context->mpicomm) == 0) { 
					zgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, 
							(BLAS_Complex16 *)alpha, (BLAS_Complex16 *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (BLAS_Complex16 *)w->val[0], 
							(ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex16 *)beta, (BLAS_Complex16 *)x->val[0], 
							(ghost_blas_idx_t *)&(x->traits->nrowspadded)); 
				} else {
					zgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (BLAS_Complex16 *)alpha, (BLAS_Complex16 *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (BLAS_Complex16 *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex16 *)&zero, (BLAS_Complex16 *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded)); 
				}
			} else {
				zgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (BLAS_Complex16 *)alpha, (BLAS_Complex16 *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (BLAS_Complex16 *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex16 *)beta, (BLAS_Complex16 *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded)); 
			}

		} else {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(v->context->mpicomm) == 0) { 
					cgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (BLAS_Complex8 *)alpha, (BLAS_Complex8 *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (BLAS_Complex8 *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex8 *)beta, (BLAS_Complex8 *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				} else {
					cgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (BLAS_Complex8 *)alpha, (BLAS_Complex8 *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (BLAS_Complex8 *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex8 *)&zero, (BLAS_Complex8 *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				}
			} else {
				cgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (BLAS_Complex8 *)alpha, (BLAS_Complex8 *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (BLAS_Complex8 *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex8 *)beta, (BLAS_Complex8 *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded));
			}
		}	
	} else {
		if (v->traits->datatype & GHOST_BINCRS_DT_DOUBLE) {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(v->context->mpicomm) == 0) { 
					dgemm(transpose,"N", (ghost_blas_idx_t *)&m,(ghost_blas_idx_t *) &n, (ghost_blas_idx_t *)&k, (double *)alpha, (double *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (double *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (double *)beta, (double *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				} else {
					dgemm(transpose,"N",(ghost_blas_idx_t *) &m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (double *)alpha, (double *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (double *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (double *)&zero, (double *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				}
			} else {
				dgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (double *)alpha, (double *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (double *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (double *)beta, (double *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded));
			}
		} else {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(v->context->mpicomm) == 0) { 
					sgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (float *)alpha, (float *)v->val[0],(ghost_blas_idx_t *) &(v->traits->nrowspadded), (float *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (float *)beta, (float *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				} else {
					sgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (float *)alpha, (float *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (float *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (float *)&zero, (float *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				}
			} else {
				sgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (float *)alpha, (float *)v->val[0], (ghost_blas_idx_t *)&(v->traits->nrowspadded), (float *)w->val[0], (ghost_blas_idx_t *)&(w->traits->nrowspadded), (float *)beta, (float *)x->val[0], (ghost_blas_idx_t *)&(x->traits->nrowspadded));
			}
		}	
	}

#ifdef GHOST_HAVE_MPI // TODO get rid of for loops
	int i,j;
	if (reduce == GHOST_GEMM_NO_REDUCE) {
		return GHOST_SUCCESS;
	} else if (reduce == GHOST_GEMM_ALL_REDUCE) {
		for (i=0; i<x->traits->nvecs; ++i) {
			for (j=0; j<x->traits->nrows; ++j) {
				MPI_safecall(MPI_Allreduce(MPI_IN_PLACE,((char *)(x->val[0]))+(i*x->traits->nrowspadded+j)*ghost_sizeofDataType(x->traits->datatype),1,ghost_mpi_dataType(x->traits->datatype),ghost_mpi_op_sum(x->traits->datatype),v->context->mpicomm));

			}
		}
	} else {
		for (i=0; i<x->traits->nvecs; ++i) {
			for (j=0; j<x->traits->nrows; ++j) {
				if (ghost_getRank(v->context->mpicomm) == reduce) {
					MPI_safecall(MPI_Reduce(MPI_IN_PLACE,((char *)(x->val[0]))+(i*x->traits->nrowspadded+j)*ghost_sizeofDataType(x->traits->datatype),1,ghost_mpi_dataType(x->traits->datatype),ghost_mpi_op_sum(x->traits->datatype),reduce,v->context->mpicomm));
				} else {
					MPI_safecall(MPI_Reduce(((char *)(x->val[0]))+(i*x->traits->nrowspadded+j)*ghost_sizeofDataType(x->traits->datatype),NULL,1,ghost_mpi_dataType(x->traits->datatype),ghost_mpi_op_sum(x->traits->datatype),reduce,v->context->mpicomm));
				}
			}
		}
	}
#else
	UNUSED(reduce);
#endif

	return GHOST_SUCCESS;

}

void ghost_mpi_add_c(ghost_mpi_c *invec, ghost_mpi_c *inoutvec, int *len)
{
	int i;
	ghost_mpi_c c;

	for (i=0; i<*len; i++, invec++, inoutvec++){
		c.x = invec->x + inoutvec->x;
		c.y = invec->y + inoutvec->y;
		*inoutvec = c;
	}
}

void ghost_mpi_add_z(ghost_mpi_z *invec, ghost_mpi_z *inoutvec, int *len)
{
	int i;
	ghost_mpi_z c;

	for (i=0; i<*len; i++, invec++, inoutvec++){
		c.x = invec->x + inoutvec->x;
		c.y = invec->y + inoutvec->y;
		*inoutvec = c;
	}
}
