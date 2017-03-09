#ifndef GHOST_COMPATIBILITY_CHECK_H
#define GHOST_COMPATIBILITY_CHECK_H

#include "context.h"
#include "sparsemat.h"
#include "densemat.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct
{
  ghost_sparsemat *mat;

  ghost_densemat *left1;
  ghost_densemat *left2;
  ghost_densemat *left3;
  ghost_densemat *left4;

  ghost_densemat *right1;
  ghost_densemat *right2;
  ghost_densemat *right3;
  ghost_densemat *right4;
}
ghost_compatible_mat_vec;


typedef struct
{
  ghost_densemat *A;
  ghost_densemat *B;
  ghost_densemat *C;
  ghost_densemat *D;
  ghost_densemat *OUT;

  char transA;
  char transB;
  char transC;
  char transD;
}
ghost_compatible_vec_vec;

typedef struct
{
  ghost_densemat *IN_A;
  ghost_densemat *IN_B;
  ghost_densemat *OUT_A;
  ghost_densemat *OUT_B;
}
ghost_compatible_vec_init;


//Checks the compatibility for routines involving sparsematrix and densematrix
ghost_error ghost_check_mat_vec_compatibility(ghost_compatible_mat_vec *data, ghost_context *ctx);

//Checks the compatibility for routines involving only densematrix
ghost_error ghost_check_vec_vec_compatibility(ghost_compatible_vec_vec *data);

//Checks the compatibility for routines involving initialization (or
//overwriting) of densematrix
ghost_error ghost_check_vec_init_compatibility(ghost_compatible_vec_init *data);



#ifdef __cplusplus
}
#endif

extern const ghost_compatible_mat_vec GHOST_COMPATIBLE_MAT_VEC_INITIALIZER;
extern const ghost_compatible_vec_vec GHOST_COMPATIBLE_VEC_VEC_INITIALIZER;
extern const ghost_compatible_vec_init GHOST_COMPATIBLE_VEC_INIT_INITIALIZER;

#endif
