/** 
 * @file constants.h
 * @brief Constant defitions.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CONSTANTS_H
#define GHOST_CONSTANTS_H

/**
 * @brief This is the alignment size for memory allocated using `ghost_malloc_align()`.
 */
#define GHOST_DATA_ALIGNMENT 64

/**
 * @brief The maximum unrolling factor for rows.
 * This influences the padding of dense matrices.
 */
#define GHOST_MAX_ROWS_UNROLL 8

#endif

