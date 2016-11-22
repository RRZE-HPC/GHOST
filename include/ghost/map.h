/**
 * @file map.h
 * @brief Types and functions related to GHOST maps.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_MAP_H
#define GHOST_MAP_H

#include "config.h"
#include "types.h"
#include "sparsemat_src.h"

/**
 * @brief Possible types of maps.
 */
typedef enum
{
    /**
     * @brief No type associated yet.
     */
    GHOST_MAP_NONE,
    /**
     * @brief A row map.
     */
    GHOST_MAP_ROW,
    /**
     * @brief A column map.
     */
    GHOST_MAP_COL
} 
ghost_maptype;

/**
 * @brief Possible sources of a sparse matrix. 
 */
typedef enum {
    /**
     * @brief The matrix comes from a binary CRS file.
     */
    GHOST_SPARSEMAT_SRC_FILE,
    /**
     * @brief The matrix comes from a Matrix Market file.
     */
    GHOST_SPARSEMAT_SRC_MM,
    /**
     * @brief The matrix is generated by a function.
     */
    GHOST_SPARSEMAT_SRC_FUNC,
    /**
     * @brief Empty source.
     */
    GHOST_SPARSEMAT_SRC_NONE
} ghost_sparsemat_src;

/**
 * @brief Possible distribution criteria of maps.
 */
typedef enum {
    /**
     * @brief Distribute by number of non zero entries.
     */
    GHOST_MAP_DIST_NNZ,
    /**
     * @brief Distribute by number of rows.
     */
    GHOST_MAP_DIST_NROWS
} ghost_map_dist_type;
    
/**
 * @brief Possible flags to maps.
 */
typedef enum {
    GHOST_MAP_DEFAULT=0,
    /**
    * @brief Does not make a distinction between local and remote entries if set; this might lead to higher communication time
    */
    GHOST_PERM_NO_DISTINCTION=1,

} ghost_map_flags;

/**
 * @brief A GHOST map.
 */
typedef struct 
{
    /**
     * @brief The global dimension.
     */
    ghost_gidx gdim;
    /**
     * @brief The offset into ::gdim of each rank.
     */
    ghost_gidx *goffs;
    /**
     * @brief The offset into ::gdim for this rank.
     */
    ghost_gidx offs;
    /**
     * @brief The local dimension of each rank.
     */
    ghost_lidx *ldim;
    /**
     * @brief The local dimension for this rank.
     */
    ghost_lidx dim;
    /**
     * @brief The local dimension including halo elements for this rank.
     */
    ghost_lidx dimhalo;
    /**
     * @brief The local dimension including halo elements and padding for this rank.
     */
    ghost_lidx dimpad;
    /**
     * @brief The number of halo elements.
     */
    ghost_lidx nhalo;
    /**
     * @brief The local permutation 
     */
    ghost_lidx *loc_perm;
    /**
     * @brief The local inverse permutation.
     */
    ghost_lidx *loc_perm_inv;
    /**
     * @brief The global permutation.
     */
    ghost_gidx *glb_perm;
    /**
     * @brief The global inverse permutation.
     */
    ghost_gidx *glb_perm_inv;
    /**
     * @brief The local permutation in CUDA memory. 
     */
    ghost_lidx *cu_loc_perm;
    /**
     * @brief The map's type.
     */
    ghost_maptype type;
    /**
     * @brief The associated MPI communicator.
     */
    ghost_mpi_comm mpicomm;
    /**
     * @brief The map's flags.
     */
    ghost_map_flags flags;
    /**
     if the reference counter is 0, ghost_map_destroy deletes the 
     data structure. Otherwise, it just decreases the reference count.
     Whenever the map is passed to a densemat or other ghost object,  
     its reference count is incremented. Such objects should call 
     ghost_map_destroy when deleted themselves.
    */
    int ref_count;
} 
ghost_map;

#ifdef __cplusplus
extern "C" {
#endif
    /**
     * @brief Initialize a map's distribution.
     *
     * @param map The map.
     * @param matsrc The sparse matrix construction function or NULL.
     * @param weight The weight of this rank.
     * @param distType The distribution scheme.
     * @param el_per_rank An array of length $nranks which prescribed the distribution or NULL.
     *
     * In case matsrc is NULL, the only possible distType is ::GHOST_MAP_DIST_NROWS.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_map_create_distribution(ghost_map *map, ghost_sparsemat_src_rowfunc *matsrc, double weight, ghost_map_dist_type distType,ghost_lidx *el_per_rank);
    ghost_error ghost_map_create(ghost_map **map, ghost_gidx gdim, ghost_mpi_comm comm, ghost_maptype type, ghost_map_flags flags);
    /**
     * @brief Create a light map with only a dimension and an MPI communicator.
     *
     * This map is usually used for non-distributed dense matrices.
     * The map does not have to be free'd by the user but will be free'd in ghost_densemat_destroy().
     *
     * @param dim The dimension.
     * @param mpicomm The MPI communicator.
     *
     * @return The map.
     */
    ghost_map *ghost_map_create_light(ghost_lidx dim, ghost_mpi_comm mpicomm);
    void ghost_map_destroy(ghost_map *map);
    /**
     * @brief Get the rank of a given global row in a given map.
     *
     * @param map The map.
     * @param row The global row.
     *
     * @return The rank inside the map's MPI communicator which owns the given row.
     */
    int ghost_rank_of_row(ghost_map *map, ghost_gidx row);
#ifdef __cplusplus
}
#endif

#endif
