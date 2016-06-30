/**
 * @file datatransfers.h
 * @brief Functions for tracking data transfers in a parallel run.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_DATATRANSFERS_H
#define GHOST_DATATRANSFERS_H
#include "error.h"

/**
 * @brief Indicate that a data transfers is done to a GPU instead of another
 * MPI rank.
 */
#define GHOST_DATATRANSFER_RANK_GPU -1
/**
 * @brief Regard data transfers to any rank.
 */
#define GHOST_DATATRANSFER_RANK_ALL -2
/**
 * @brief Regard data transfers to any rank and the process-local GPU.
 */
#define GHOST_DATATRANSFER_RANK_ALL_W_GPU -3

/**
 * @brief The direction of a data transfer.
 */
typedef enum {
    /**
     * @brief Incoming data transfer.
     */
    GHOST_DATATRANSFER_IN,
    /**
     * @brief Outgoing data transfer.
     */
    GHOST_DATATRANSFER_OUT,
    /**
     * @brief Any direction.
     */
    GHOST_DATATRANSFER_ANY
} ghost_datatransfer_direction_t;

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Register a data transfer.
     *
     * @param tag The tag of the data transfer.
     * @param dir The transfer's direction.
     * @param rank The rank of the communication partner.
     * @param volume The size of the data transfer.
     *
     * @return 
     */
    ghost_error ghost_datatransfer_register(const char *tag, ghost_datatransfer_direction_t dir, int rank, size_t volume);
    /**
     * @brief Create a string summarizing all data transfers.
     *
     * @param str Where to store the string.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_datatransfer_summarystring(char **str);
    size_t ghost_datatransfer_volume_get(const char *tag, ghost_datatransfer_direction_t dir, int rank);

#ifdef __cplusplus
}
#endif

#endif
