#ifndef __BJDS_CU_KERNEL_H__
#define __BJDS_CU_KERNEL_H__

#include <ghost.h>

void c_BJDS_kernel_wrap(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
void d_BJDS_kernel_wrap(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
void s_BJDS_kernel_wrap(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
void z_BJDS_kernel_wrap(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);

#endif

