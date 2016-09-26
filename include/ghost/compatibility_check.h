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

//Checks the compatibility for routines involving sparsematrix and densematrix
ghost_error ghost_check_mat_vec_compatibility(ghost_compatible_mat_vec *data, ghost_context *ctx);

//Checks the compatibility for routines involving only densematrix
ghost_error ghost_check_vec_vec_compatibility(ghost_compatible_vec_vec *data, ghost_context *ctx);


//Internal functions
bool checkLeft(ghost_densemat *left, ghost_context *ctx);
bool checkRight(ghost_densemat *right, ghost_context *ctx);
bool makeSimilar(ghost_densemat *vec1, ghost_densemat *vec2, ghost_context *ctx);
bool clonePermProperties(ghost_densemat *vec1, ghost_densemat_permuted vec2_row, ghost_densemat_permuted vec2_col, bool permuted_vec2 , ghost_context *ctx);


#ifdef __cplusplus
}
#endif

extern const ghost_compatible_mat_vec GHOST_COMPATIBLE_MAT_VEC_INITIALIZER;
extern const ghost_compatible_vec_vec GHOST_COMPATIBLE_VEC_VEC_INITIALIZER;


#endif
