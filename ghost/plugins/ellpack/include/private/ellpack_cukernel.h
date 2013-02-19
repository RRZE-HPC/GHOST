#ifndef __ELLPACK_CU_KERNEL_H__
#define __ELLPACK_CU_KERNEL_H__

#include <ghost.h>

void c_ELLPACK_kernel_wrap(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
void d_ELLPACK_kernel_wrap(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
void s_ELLPACK_kernel_wrap(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
void z_ELLPACK_kernel_wrap(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);

#endif

