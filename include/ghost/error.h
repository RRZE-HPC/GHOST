/**
 * @file error.h
 * @brief Types, functions and macros for error handling.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_ERROR_H
#define GHOST_ERROR_H

#ifdef __cplusplus
#include <cstring>
#else
#include <string.h>
#endif
#include "log.h"

/**
 * @brief Error return type.
 */
typedef enum {
   /**
    * @brief No error occured.
    */
    GHOST_SUCCESS,
    GHOST_ERR_INVALID_ARG,
    GHOST_ERR_MPI,
    GHOST_ERR_CUDA,
    GHOST_ERR_CUBLAS,
    GHOST_ERR_CURAND,
    GHOST_ERR_HWLOC,
    GHOST_ERR_SCOTCH,
    GHOST_ERR_UNKNOWN,
    GHOST_ERR_NOT_IMPLEMENTED,
    GHOST_ERR_IO
} ghost_error_t;

/**
 * @brief This macro should be used for calling a GHOST function inside
 * a function which itself returns a ghost_error_t.
 * It calls the function and in case of an error logs the error message
 * and returns the according ghost_error_t which was return by the called function.
 *
 * @param call The complete function call.
 *
 * @return A ghost_error_t in case the function return an error.
 */
#define GHOST_CALL_RETURN(call) {\
    ghost_error_t err = GHOST_SUCCESS;\
    GHOST_CALL(call,err);\
    if (err != GHOST_SUCCESS) {\
        return err;\
    }\
}\

/**
 * @brief This macro should be used for calling a GHOST function inside
 * a function which itself returns a ghost_error_t but needs to do some clean up before returning..
 * It calls the function. In case of an error it logs the error message, sets the parameter __err
 * to the occured ghost_error_t and jumps (goto) to the defined parameter label.
 *
 * @param call The function call.
 * @param label The jump label where clean up is performed.
 * @param __err A defined ghost_error_t variable which will be set to the error returned by the calling function.
 */
#define GHOST_CALL_GOTO(call,label,__err) {\
    GHOST_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

/**
 * @brief This macro should be used for calling a GHOST function inside
 * another function. The parameter __err will be set to the error which occured in the call.
 * This macro is probably not very useful by itself, cf. @GHOST_CALL_GOTO and @GHOST_CALL_RETURN instead.
 *
 * @param call The function call.
 * @param __err A defined ghost_error_t variable which will be set to the error returned by the calling function.
 */
#define GHOST_CALL(call,__err) {\
    __err = call;\
    if (__err != GHOST_SUCCESS) {\
        LOG(GHOST_ERROR,ANSI_COLOR_RED,"%s",ghost_error_string((ghost_error_t)__err));\
    }\
}\

#define MPI_CALL_RETURN(call) {\
    ghost_error_t ret = GHOST_SUCCESS;\
    MPI_CALL(call,ret);\
    if (ret != GHOST_SUCCESS) {\
        return ret;\
    }\
}\

#define MPI_CALL_GOTO(call,label,__err) {\
    MPI_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define MPI_CALL(call,__err) {\
    int err = call;\
    if (err != MPI_SUCCESS) {\
        char errstr[MPI_MAX_ERROR_STRING];\
        int strlen;\
        MPI_Error_string(err,errstr,&strlen);\
        ERROR_LOG("MPI Error: %s",errstr);\
        __err = GHOST_ERR_MPI;\
    }\
}\

#define CUDA_CALL_RETURN(call) {\
    ghost_error_t ret = GHOST_SUCCESS;\
    CUDA_CALL(call,ret);\
    if (ret != GHOST_SUCCESS) {\
        return ret;\
    }\
}\

#define CUDA_CALL_GOTO(call,label,__err) {\
    CUDA_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define CUDA_CALL(call,__err) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        ERROR_LOG("CUDA Error: %s",cudaGetErrorString(err));\
        __err = GHOST_ERR_CUDA;\
    }\
}\

#define CUBLAS_CALL_RETURN(call) {\
    ghost_error_t ret = GHOST_SUCCESS;\
    CUBLAS_CALL(call,ret);\
    if (ret != GHOST_SUCCESS) {\
        return ret;\
    }\
}\

#define CUBLAS_CALL_GOTO(call,label,__err) {\
    CUBLAS_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define CUBLAS_CALL(call,__err) {\
    cublasStatus_t err = call;\
    if (err != CUBLAS_STATUS_SUCCESS) {\
        ERROR_LOG("CUBLAS Error: %d",err);\
        __err = GHOST_ERR_CUBLAS;\
    }\
}\

#define CURAND_CALL_RETURN(call) {\
    ghost_error_t ret = GHOST_SUCCESS;\
    CURAND_CALL(call,ret);\
    if (ret != GHOST_SUCCESS) {\
        return ret;\
    }\
}\

#define CURAND_CALL_GOTO(call,label,__err) {\
    CURAND_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define CURAND_CALL(call,__err) {\
    curandStatus_t err = call;\
    if (err != CURAND_STATUS_SUCCESS) {\
        ERROR_LOG("CURAND Error: %d",err);\
        __err = GHOST_ERR_CURAND;\
    }\
}\

#define SCOTCH_CALL_RETURN(call) {\
    ghost_error_t ret = GHOST_SUCCESS;\
    SCOTCH_CALL(call,ret);\
    if (ret != GHOST_SUCCESS) {\
        return ret;\
    }\
}\

#define SCOTCH_CALL_GOTO(call,label,__err) {\
    SCOTCH_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define SCOTCH_CALL(call,__err) {\
    int err = call;\
    if (err) {\
        ERROR_LOG("SCOTCH Error: %d",err);\
        __err = GHOST_ERR_SCOTCH;\
    }\
}\

#ifdef __cplusplus
extern "C" {
#endif

    char * ghost_error_string(ghost_error_t e);

#ifdef __cplusplus
}
#endif

#endif
