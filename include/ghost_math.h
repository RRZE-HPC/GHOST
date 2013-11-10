#ifndef __GHOST_MATH_H__
#define __GHOST_MATH_H__


void ghost_normalizeVec(ghost_vec_t *);
void ghost_dotProduct(ghost_vec_t *, ghost_vec_t *, void *);
int ghost_spmvm(ghost_context_t *context, ghost_vec_t *res, ghost_mat_t *mat, ghost_vec_t *invec, 
		int *spmvmOptions);
int ghost_gemm(char *, ghost_vec_t *,  ghost_vec_t *, ghost_vec_t *, void *, void *, int); 
void ghost_mpi_add_c(ghost_mpi_c *invec, ghost_mpi_c *inoutvec, int *len);
void ghost_mpi_add_z(ghost_mpi_z *invec, ghost_mpi_z *inoutvec, int *len);
#endif

