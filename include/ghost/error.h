/**
 * @file error.h
 * @brief Types, functions and macros for error handling.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_ERROR_H
#define GHOST_ERROR_H

#include <string.h>
#include "log.h"

#ifndef GHOST_HAVE_ZOLTAN
#define ZOLTAN_OK 0
#endif

/**
 * @brief Error return type.
 */
typedef enum {
    /**
     * @brief No error occured.
     */
    GHOST_SUCCESS,
    /**
     * @brief One or more of the arguments are invalid.
     */
    GHOST_ERR_INVALID_ARG,
    /**
     * @brief An error in an MPI call occured.
     */
    GHOST_ERR_MPI,
    /**
     * @brief An error in a CUDA call occured.
     */
    GHOST_ERR_CUDA,
    /**
     * @brief An error in a CUBLAS call occured.
     */
    GHOST_ERR_CUBLAS,
    /**
     * @brief An error in a CUSPARSE call occured.
     */
    GHOST_ERR_CUSPARSE,
    /**
     * @brief An error in a CURAND call occured.
     */
    GHOST_ERR_CURAND,
    /**
     * @brief An error in a Hwloc call occured.
     */
    GHOST_ERR_HWLOC,
    /**
     * @brief An error in a SCOTCH call occured.
     */
    GHOST_ERR_SCOTCH,
    /**
     * @brief An error in a Zoltan call occured.
     */
    GHOST_ERR_ZOLTAN,
    /**
     * @brief An unknown error occured.
     */
    GHOST_ERR_UNKNOWN,
    /**
     * @brief The function is not (yet) implemented.
     */
    GHOST_ERR_NOT_IMPLEMENTED,
    /**
     * @brief An error during I/O occured.
     */
    GHOST_ERR_IO,
    /**
     * @brief An error with datatypes occured.
     */
    GHOST_ERR_DATATYPE,
    /**
     * @brief An error in a ColPack call occured.
     */
    GHOST_ERR_COLPACK,
    /**
     * @brief An error in a LAPACK call occured.
     */
    GHOST_ERR_LAPACK,
    /**
     * @brief An error in a BLAS call occured.
     */
    GHOST_ERR_BLAS
} ghost_error;

#include "errorhandler.h"

/**
 * @brief This macro should be used for calling a GHOST function inside
 * a function which itself returns a ghost_error.
 * It calls the function and in case of an error logs the error message
 * and returns the according ghost_error which was return by the called function.
 *
 * @param call The complete function call.
 *
 * @return A ghost_error in case the function return an error.
 */
#define GHOST_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    GHOST_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
    }\
}\

/**
 * @brief This macro should be used for calling a GHOST function inside
 * a function which itself returns a ghost_error but needs to do some clean up before returning..
 * It calls the function. In case of an error it logs the error message, sets the parameter __err
 * to the occured ghost_error and jumps (goto) to the defined parameter label.
 *
 * @param call The function call.
 * @param label The jump label where clean up is performed.
 * @param __err A defined ghost_error variable which will be set to the error returned by the calling function.
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
 * This macro is probably not very useful by itself, cf. #GHOST_CALL_GOTO and #GHOST_CALL_RETURN instead.
 *
 * @param call The function call.
 * @param __err A defined ghost_error variable which will be set to the error returned by the calling function.
 */
#define GHOST_CALL(call,__err) {\
    __err = call;\
    if (__err != GHOST_SUCCESS) {\
        LOG(GHOST_ERROR,ANSI_COLOR_RED,"%s",ghost_error_string((ghost_error)__err));\
    }\
}\

#define MPI_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    MPI_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
    }\
}\

#define MPI_CALL_GOTO(call,label,__err) {\
    MPI_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define MPI_CALL(call,__err) {\
    int mpicallmacroerr = call;\
    if (mpicallmacroerr != MPI_SUCCESS) {\
        char errstr[MPI_MAX_ERROR_STRING];\
        int strlen;\
        MPI_Error_string(mpicallmacroerr,errstr,&strlen);\
        ERROR_LOG("MPI Error: %s",errstr);\
        ghost_errorhandler h = ghost_errorhandler_get(GHOST_ERR_MPI);\
        if (h) {\
            h((void *)&mpicallmacroerr);\
        }\
        __err = GHOST_ERR_MPI;\
    } else {\
        __err = GHOST_SUCCESS;\
    }\
}\

#define HWLOC_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    HWLOC_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
    }\
}\

#define HWLOC_CALL_GOTO(call,label,__err) {\
    HWLOC_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define HWLOC_CALL(call,__err) {\
    int __hwlocerr = call;\
    if (__hwlocerr) {\
        ERROR_LOG("HWLOC Error: %d",__hwlocerr);\
        ghost_errorhandler h = ghost_errorhandler_get(GHOST_ERR_HWLOC);\
        if (h) {\
            h((void *)&__hwlocerr);\
        }\
        __err = GHOST_ERR_HWLOC;\
    } else {\
        __err = GHOST_SUCCESS;\
    }\
}\

#define CUDA_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    CUDA_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
    }\
}\

#define CUDA_CALL_GOTO(call,label,__err) {\
    CUDA_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define CUDA_CALL(call,__err) {\
    cudaError_t __cuerr = call;\
    if (__cuerr != cudaSuccess) {\
        ERROR_LOG("CUDA Error: %s (%d)",cudaGetErrorString(__cuerr),(int)__cuerr);\
        ghost_errorhandler h = ghost_errorhandler_get(GHOST_ERR_CUDA);\
        if (h) {\
            h((void *)&__cuerr);\
        }\
        __err = GHOST_ERR_CUDA;\
    } else {\
        __err = GHOST_SUCCESS;\
    }\
}\

#define CUBLAS_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    CUBLAS_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
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
        switch (err) {\
            case CUBLAS_STATUS_NOT_INITIALIZED:\
                ERROR_LOG("CUBLAS error: CUBLAS_STATUS_NOT_INITIALIZED");\
                break;\
            case CUBLAS_STATUS_ALLOC_FAILED:\
                ERROR_LOG("CUBLAS error: CUBLAS_STATUS_ALLOC_FAILED");\
                break;\
            case CUBLAS_STATUS_INVALID_VALUE:\
                ERROR_LOG("CUBLAS error: CUBLAS_STATUS_INVALID_VALUE");\
                break;\
            case CUBLAS_STATUS_ARCH_MISMATCH:\
                ERROR_LOG("CUBLAS error: CUBLAS_STATUS_ARCH_MISMATCH");\
                break;\
            case CUBLAS_STATUS_MAPPING_ERROR:\
                ERROR_LOG("CUBLAS error: CUBLAS_STATUS_MAPPING_ERROR");\
                break;\
            case CUBLAS_STATUS_EXECUTION_FAILED:\
                ERROR_LOG("CUBLAS error: CUBLAS_STATUS_EXECUTION_FAILED");\
                break;\
            case CUBLAS_STATUS_INTERNAL_ERROR:\
                ERROR_LOG("CUBLAS error: CUBLAS_STATUS_INTERNAL_ERROR");\
                break;\
            case CUBLAS_STATUS_NOT_SUPPORTED:\
                ERROR_LOG("CUBLAS error: CUBLAS_STATUS_NOT_SUPPORTED");\
                break;\
            case CUBLAS_STATUS_LICENSE_ERROR:\
                ERROR_LOG("CUBLAS error: CUBLAS_STATUS_LICENSE_ERROR");\
                break;\
            default:\
                ERROR_LOG("CUBLAS error: Unknown CUBLAS error");\
                break;\
        }\
        ghost_errorhandler h = ghost_errorhandler_get(GHOST_ERR_CUBLAS);\
        if (h) {\
            h((void *)&err);\
        }\
        __err = GHOST_ERR_CUBLAS;\
    } else {\
        __err = GHOST_SUCCESS;\
    }\
}\

#define CURAND_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    CURAND_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
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
        ERROR_LOG("CURAND Error: %d",(int)err);\
        ghost_errorhandler h = ghost_errorhandler_get(GHOST_ERR_CURAND);\
        if (h) {\
            h((void *)&err);\
        }\
        __err = GHOST_ERR_CURAND;\
    } else {\
        __err = GHOST_SUCCESS;\
    }\
}\

#define CUSPARSE_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    CUSPARSE_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
    }\
}\

#define CUSPARSE_CALL_GOTO(call,label,__err) {\
    CUSPARSE_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define CUSPARSE_CALL(call,__err) {\
    cusparseStatus_t err = call;\
    if (err != CUSPARSE_STATUS_SUCCESS) {\
        ERROR_LOG("CUSPARSE Error: %d",(int)err);\
        ghost_errorhandler h = ghost_errorhandler_get(GHOST_ERR_CUSPARSE);\
        if (h) {\
            h((void *)&err);\
        }\
        __err = GHOST_ERR_CUSPARSE;\
    } else {\
        __err = GHOST_SUCCESS;\
    }\
}\

#define SCOTCH_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    SCOTCH_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
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
        ghost_errorhandler h = ghost_errorhandler_get(GHOST_ERR_SCOTCH);\
        if (h) {\
            h((void *)&err);\
        }\
        __err = GHOST_ERR_SCOTCH;\
    } else {\
        __err = GHOST_SUCCESS;\
    }\
}\

#define COLPACK_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    COLPACK_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
    }\
}\

#define COLPACK_CALL_GOTO(call,label,__err) {\
    COLPACK_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define COLPACK_CALL(call,__err) {\
    int err = call;\
    if (err != _TRUE) {\
        ERROR_LOG("ColPack Error: %d",err);\
        ghost_errorhandler h = ghost_errorhandler_get(GHOST_ERR_COLPACK);\
        if (h) {\
            h((void *)&err);\
        }\
        __err = GHOST_ERR_COLPACK;\
    } else {\
        __err = GHOST_SUCCESS;\
    }\
}\

#define ZOLTAN_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    ZOLTAN_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
    }\
}\

#define ZOLTAN_CALL_GOTO(call,label,__err) {\
    ZOLTAN_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define ZOLTAN_CALL(call,__err) {\
    int err = call;\
    if (err != ZOLTAN_OK) {\
        ERROR_LOG("Zoltan Error: %d",err);\
        ghost_errorhandler h = ghost_errorhandler_get(GHOST_ERR_ZOLTAN);\
        if (h) {\
            h((void *)&err);\
        }\
        __err = GHOST_ERR_ZOLTAN;\
    } else {\
        __err = GHOST_SUCCESS;\
    }\
}\

#define BLAS_CALL_RETURN(call) {\
    ghost_error macroerr = GHOST_SUCCESS;\
    BLAS_CALL(call,macroerr);\
    if (macroerr != GHOST_SUCCESS) {\
        return macroerr;\
    }\
}\

#define BLAS_CALL_GOTO(call,label,__err) {\
    BLAS_CALL(call,__err);\
    if (__err != GHOST_SUCCESS) {\
        goto label;\
    }\
}\

#define BLAS_CALL(call,__err) {\
    call;\
    int err = ghost_blas_err_pop();\
    if (err) {\
        ERROR_LOG("BLAS Error: %d",err);\
        ghost_errorhandler h = ghost_errorhandler_get(GHOST_ERR_BLAS);\
        if (h) {\
            h((void *)&err);\
        }\
        __err = GHOST_ERR_BLAS;\
    } else {\
        __err = GHOST_SUCCESS;\
    }\
}\

#ifdef __cplusplus
    extern "C" {
#endif

        /**
         * @ingroup stringification
         *
         * @brief Get a string of the GHOST error. 
         *
         * @param e The error.
         *
         * @return The string.
         */
        char * ghost_error_string(ghost_error e);

#ifdef __cplusplus
    }
#endif

#endif
