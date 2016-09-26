#include "ghost/compatibility_check.h"
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/math.h"

const ghost_compatible_mat_vec GHOST_COMPATIBLE_MAT_VEC_INITIALIZER = {
  .mat    = NULL,
  .left1  = NULL,
  .left2  = NULL,
  .left3  = NULL,
  .left4  = NULL,
  .right1 = NULL,
  .right2 = NULL,
  .right3 = NULL,
  .right4 = NULL,
};

//checks OUT = A^(transa) * B^(transb) + C^(transc) + D^(transd)
const ghost_compatible_vec_vec GHOST_COMPATIBLE_VEC_VEC_INITIALIZER = {
  .A = NULL,
  .B = NULL,
  .C = NULL,
  .D = NULL,
  .OUT = NULL,
  .transA = 'N',
  .transB = 'N',
  .transC = 'N',
  .transD = 'N'
};


bool checkLeft(ghost_densemat *left, ghost_context *ctx)
{
  bool flag = true;
  if(left->traits.permutemethod != ROW)
  {
    #ifndef GHOST_COMPATIBLE_PERM
    WARNING_LOG("Left Vector not in Row space")
    flag = false;
    #else
    if(left->traits.permutemethod == NONE)
    {
     convert_permutation_method(left, ctx, ROW, true); 
      //maybe need to convert to row perm
    }
    else if(left->traits.permutemethod == COLUMN)
    {
      switch_permutation_method(left, ctx, true); 
    }
    #endif
  }
  return flag;
}

bool checkRight(ghost_densemat *right, ghost_context *ctx)
{
  bool flag = true;
  if(right->traits.permutemethod != COLUMN)
  {
    #ifndef GHOST_COMPATIBLE_PERM
    WARNING_LOG("Right Vector not in Column space")
    flag = false;
    #else
    if(right->traits.permutemethod == NONE)
    {
     convert_permutation_method(right, ctx, COLUMN, true); 
      //maybe need to convert to row perm
    }
    else if(right->traits.permutemethod == ROW)
    {
      switch_permutation_method(right, ctx, true); 
    }
    #endif
  }
return flag;
}
    
//make vec2 similar to vec1
bool makeSimilar(ghost_densemat *vec1, ghost_densemat *vec2, ghost_context *ctx)
{
  bool flag = true;
  ghost_densemat_permuted vec1_dim, vec2_dim;
  vec1_dim = vec1->traits.permutemethod;
  vec2_dim = vec2->traits.permutemethod;
 
  bool permuted_vec1=false, permuted_vec2=false;

  permuted_vec1 = vec1->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED;
  permuted_vec2 = vec2->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED;
   
  if( (vec1_dim != vec2_dim) || (permuted_vec1 != permuted_vec2)  ) 
  {
    flag = false;
    if(vec1_dim != NONE) 
    {
     if(vec2_dim == NONE) 
      {
       convert_permutation_method(vec2, ctx, vec1_dim, permuted_vec1);
      } 
      else if(permuted_vec1 != permuted_vec2) 
      {
         if(permuted_vec1) 
          {
            switch_permutation_method(vec2, ctx, true); 
          } 
          else 
          {
            vec2->permute(vec2,ctx,GHOST_PERMUTATION_PERM2ORIG); 
            switch_permutation_method(vec2, ctx, false); 
          }
      }
      else 
      {
         switch_permutation_method(vec2, ctx, false);
      }  
    } 
    else if(vec2_dim != NONE)
    {   
      convert_permutation_method(vec1, ctx, vec2_dim, permuted_vec2);
    }    
  }

  if(flag == false)
  { 
    #ifndef GHOST_COMPATIBLE_PERM
    WARNING_LOG("DENSEMAT not in same space")
    #endif
  }
  return flag; 
}

//make vec1 satisfy the map specified in argument
bool clonePermProperties(ghost_densemat *vec1, ghost_densemat_permuted vec2_row, ghost_densemat_permuted vec2_col, bool permuted_vec2 , ghost_context *ctx)
{
  bool flag = true;
  if(vec2_col != NONE) 
  {
    flag = false;
    ERROR_LOG("COMPATIBILITY CHECK: Transposed densemat cannot be stored")
  }
  else 
  { 
      ghost_densemat_permuted vec1_dim, vec2_dim;
     
      vec1_dim = vec1->traits.permutemethod;
      vec2_dim = vec2_row;

      bool permuted_vec1=false;

      permuted_vec1 = vec1->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED;

      if( (vec1_dim != vec2_dim) || (permuted_vec1 != permuted_vec2)  ) 
      {
        flag = false;
        if(vec2_dim != NONE) 
        {
          if(vec1_dim == NONE) 
          {
            convert_permutation_method(vec1, ctx, vec2_dim, permuted_vec2);
          } 
          else if(permuted_vec1 != permuted_vec2) 
          {
              if(permuted_vec2) 
              {
                switch_permutation_method(vec1, ctx, true); 
              } 
              else 
              {
                vec1->permute(vec1,ctx,GHOST_PERMUTATION_PERM2ORIG); 
                switch_permutation_method(vec1, ctx, false); 
              }
          }
          else
          {
              switch_permutation_method(vec1, ctx, false);
          }  
        } 
        else if(vec2_dim != NONE)
        {   
          flag  = false;   
          ERROR_LOG("Cannot convert back to permutemethod=NONE")
        }    
      }
   }


  if(flag == false)
  { 
    #ifndef GHOST_COMPATIBLE_PERM
    WARNING_LOG("DENSEMAT not in same space")
    #endif
  }
  return flag; 
}

ghost_error ghost_check_mat_vec_compatibility(ghost_compatible_mat_vec *data, ghost_context *ctx)
{
  ghost_error ret = GHOST_SUCCESS;
  bool flag = true;
  
  ghost_densemat* left[4] = {data->left1, data->left2, data->left3, data->left4};
  ghost_densemat* right[4] = {data->right1, data->right2, data->right3, data->right4};

  if(data->mat != NULL)
  {
    for(int i=0; i<4; ++i) {
      if(left[i] != NULL)
      {
        flag = checkLeft(left[i], ctx);      
      }
    }
    for(int i=0; i<4; ++i) {
      if(right[i] != NULL)
      {
       flag = checkRight(right[i],ctx);      
      }
    }
  } else {
    flag = false;
    ERROR_LOG("Provide matrix for checking compatibility")
  }

 if(flag == false)
 {
  ret = GHOST_ERR_COMPATIBILITY;
 }
 
  return ret;
}


/*void print_permute_type(ghost_densemat_permuted arg)
{
  printf("%s",(arg==ROW)?"ROW":(arg==COLUMN)?"COLUMN":"NONE");
}
*/

ghost_error ghost_check_vec_vec_compatibility(ghost_compatible_vec_vec *data, ghost_context *ctx)
{
  ghost_error ret = GHOST_SUCCESS;
  bool flag = true;
  
  ghost_densemat_permuted row_E, col_E, row_A, col_A, row_B, col_B;
  bool permuted_E = false;

  if((data->A != NULL) && (data->B != NULL)) 
  {
    //assumption transposed densemat is not stored
    row_A = (data->transA=='N')?(data->A)->traits.permutemethod:NONE;
    col_A = (data->transA=='N')?NONE:(data->A)->traits.permutemethod;
 
    row_B = (data->transB=='N')?(data->B)->traits.permutemethod:NONE;
    col_B = (data->transB=='N')?NONE:(data->B)->traits.permutemethod;

    if(col_B != NONE)
    {
      ERROR_LOG("COMPATIBILITY CHECK: Transposed densemat cannot be stored")
      flag = false;
    }
    else 
    {

        row_E = row_A;
        col_E = col_B;

        bool permuted_A = false, permuted_B = false;
        if((data->A)->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED) {
          permuted_A = true;
        } 
        if((data->B)->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED) {
          permuted_B = true;
        } 

        if(col_A != row_B || ((col_A != NONE)&&(permuted_A != permuted_B)))
        {
       
          #ifndef GHOST_COMPATIBLE_PERM
          WARNING_LOG("DENSEMAT inner dimension not compatible for multiplication")
          #else
          //B is made to be compatible with A if col_A != NONE (TODO: It might or might not be optimal)
          makeSimilar(data->A, data->B, ctx); 
          #endif            
        }
        permuted_A = false;
        permuted_B = false;
        if( (row_A!= NONE) && ((data->A)->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED) ) {
         permuted_A = true;
        } 
        if( (col_B !=  NONE) && ((data->B)->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED) ) {
         permuted_B = true;
         ERROR_LOG("COMPATIBILITY CHECK : Shouldn't have been here")
        } 
        permuted_E = permuted_A | permuted_B; 

        if(data->OUT != NULL)
        {
         flag = clonePermProperties(data->OUT, row_E, col_E, permuted_E , ctx); 
        }
     }
     
  }

  if(data->C != NULL)
  {
   flag = makeSimilar(data->OUT, data->C, ctx);
  }   

  if(data->D != NULL)
  {
   flag = makeSimilar(data->OUT, data->D, ctx);
  }   
 
 if(flag == false)
 {
  ret = GHOST_ERR_COMPATIBILITY;
 }
 return ret;
}

