#include "ghost/compatibility_check.h"
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/math.h"

//Internal functions
static bool checkLeft(ghost_densemat *left);
static bool checkRight(ghost_densemat *right);
static bool makeSimilar(ghost_densemat *vec1, ghost_densemat *vec2);
static bool clonePermProperties(ghost_densemat *vec1, ghost_maptype vec2_row, ghost_maptype vec2_col, bool permuted_vec2);

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


static bool checkLeft(ghost_densemat *left)
{
    bool flag = true;
    if(left->map != left->context->row_map)
    {
#ifndef GHOST_COMPATIBLE_PERM
        WARNING_LOG("Left Vector not in Row space")
            flag = false;
#else
        if(left->map == left->context->col_map)
        {
            ghost_densemat_swap_map(left); 
        }
#endif
    }
    return flag;
}

static bool checkRight(ghost_densemat *right)
{
    bool flag = true;
    if(right->map != right->context->col_map)
    {
#ifndef GHOST_COMPATIBLE_PERM
        WARNING_LOG("Right Vector not in Column space")
            flag = false;
#else
        if(right->map == right->context->row_map)
        {
            ghost_densemat_swap_map(right); 
        }
#endif
    }
    return flag;
}

#define MAPTYPE(vec)(vec->context?vec->context->col_map == vec->map?GHOST_MAP_COL:GHOST_MAP_ROW:GHOST_MAP_DEFAULT)

//make vec2 similar to vec1
static bool makeSimilar(ghost_densemat *vec1, ghost_densemat *vec2)
{
    bool flag = true;
    ghost_maptype vec1_dim, vec2_dim;
    vec1_dim = MAPTYPE(vec1);
    vec2_dim = MAPTYPE(vec2);

    bool permuted_vec1=false, permuted_vec2=false;

    permuted_vec1 = vec1->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED;
    permuted_vec2 = vec2->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED;

    if( (vec1_dim != vec2_dim) || (permuted_vec1 != permuted_vec2)  ) 
    {
        flag = false;
        if(vec1_dim != GHOST_MAP_DEFAULT) 
        {
            if(vec2_dim == GHOST_MAP_DEFAULT) 
            {
                ghost_densemat_set_map(vec2,vec1_dim);
                if (permuted_vec1) {
                    ghost_densemat_permute(vec2,vec2->context,GHOST_PERMUTATION_ORIG2PERM);
                }
            } 
            else if(permuted_vec1 != permuted_vec2) 
            {
                if(permuted_vec1) 
                {
                    ghost_densemat_swap_map(vec2);
                    ghost_densemat_permute(vec2,vec2->context,GHOST_PERMUTATION_PERM2ORIG);
                } 
                else 
                {
                    ghost_densemat_permute(vec2,vec2->context,GHOST_PERMUTATION_PERM2ORIG);
                    ghost_densemat_swap_map(vec2); 
                }
            }
            else 
            {
                ghost_densemat_swap_map(vec2); 
            }  
        } 
        else if(vec2_dim != GHOST_MAP_DEFAULT)
        {   
            ghost_densemat_set_map(vec1,vec2_dim);
            if (permuted_vec2) {
                ghost_densemat_permute(vec1,vec1->context,GHOST_PERMUTATION_ORIG2PERM);
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

//make vec1 satisfy the map specified in argument
bool clonePermProperties(ghost_densemat *vec1, ghost_maptype vec2_row, ghost_maptype vec2_col, bool permuted_vec2)
{
    bool flag = true;
    if(vec2_col != GHOST_MAP_DEFAULT) 
    {
        flag = false;
        ERROR_LOG("COMPATIBILITY CHECK: Transposed densemat cannot be stored")
    }
    else 
    { 
        ghost_maptype vec1_dim, vec2_dim;

        vec1_dim = MAPTYPE(vec1);
        vec2_dim = vec2_row;

        bool permuted_vec1=false;

        permuted_vec1 = vec1->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED;

        if( (vec1_dim != vec2_dim) || (permuted_vec1 != permuted_vec2)  ) 
        {
            flag = false;
            if(vec2_dim != GHOST_MAP_DEFAULT) 
            {
                if(vec1_dim == GHOST_MAP_DEFAULT) 
                {
                    ghost_densemat_set_map(vec1, vec2_dim);
                    if (permuted_vec2) {
                        ghost_densemat_permute(vec1,vec1->context,GHOST_PERMUTATION_ORIG2PERM);
                    }
                } 
                else if(permuted_vec1 != permuted_vec2) 
                {
                    if(permuted_vec2) 
                    {
                        ghost_densemat_swap_map(vec1);
                        ghost_densemat_permute(vec1, vec1->context, GHOST_PERMUTATION_ORIG2PERM); 
                    } 
                    else 
                    {
                        ghost_densemat_permute(vec1,vec1->context,GHOST_PERMUTATION_PERM2ORIG); 
                        ghost_densemat_swap_map(vec1); 
                    }
                }
                else
                {
                    ghost_densemat_swap_map(vec1);
                }  
            } 
            else if(vec2_dim != GHOST_MAP_DEFAULT)
            {   
                flag  = false;   
                ERROR_LOG("Cannot convert back to GHOST_MAP_DEFAULT")
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
                flag = checkLeft(left[i]);      
            }
        }
        for(int i=0; i<4; ++i) {
            if(right[i] != NULL)
            {
                flag = checkRight(right[i]);      
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


ghost_error ghost_check_vec_vec_compatibility(ghost_compatible_vec_vec *data, ghost_context *ctx)
{
    ghost_error ret = GHOST_SUCCESS;
    bool flag = true;

    ghost_maptype row_E, col_E, row_A, col_A, row_B, col_B;
    bool permuted_E = false;

    if((data->A != NULL) && (data->B != NULL)) 
    {
        //assumption transposed densemat is not stored
        row_A = (data->transA=='N')?MAPTYPE(data->A):GHOST_MAP_DEFAULT;
        col_A = (data->transA=='N')?GHOST_MAP_DEFAULT:MAPTYPE(data->A);

        row_B = (data->transB=='N')?MAPTYPE(data->B):GHOST_MAP_DEFAULT;
        col_B = (data->transB=='N')?GHOST_MAP_DEFAULT:MAPTYPE(data->B);

        if(col_B != GHOST_MAP_DEFAULT)
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

            if(col_A != row_B || ((col_A != GHOST_MAP_DEFAULT)&&(permuted_A != permuted_B)))
            {

#ifndef GHOST_COMPATIBLE_PERM
                WARNING_LOG("DENSEMAT inner dimension not compatible for multiplication")
#else
                    //B is made to be compatible with A if col_A != GHOST_MAP_DEFAULT (TODO: It might or might not be optimal)
                    makeSimilar(data->A, data->B); 
#endif            
            }
            permuted_A = false;
            permuted_B = false;
            if( (row_A!= GHOST_MAP_DEFAULT) && ((data->A)->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED) ) {
                permuted_A = true;
            } 
            if( (col_B !=  GHOST_MAP_DEFAULT) && ((data->B)->traits.flags & (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED) ) {
                permuted_B = true;
                ERROR_LOG("COMPATIBILITY CHECK : Shouldn't have been here")
            } 
            permuted_E = permuted_A | permuted_B; 

            if(data->OUT != NULL)
            {
                flag = clonePermProperties(data->OUT, row_E, col_E, permuted_E); 
            }
        }

    }

    if(data->C != NULL)
    {
        flag = makeSimilar(data->OUT, data->C);
    }   

    if(data->D != NULL)
    {
        flag = makeSimilar(data->OUT, data->D);
    }   

    if(flag == false)
    {
        ret = GHOST_ERR_COMPATIBILITY;
    }
    return ret;
}

