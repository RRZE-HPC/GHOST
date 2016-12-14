#ifndef KACZ_HYPRID_SPLITTING_H
#define KACZ_HYBRID_SPLITTING_H

#include "sparsemat.h"
#include "util.h"
#include "locality.h"
#include "bincrs.h"
#include "omp.h"
#include <omp.h>
#include <math.h>


ghost_error find_zone_extrema(ghost_sparsemat *mat, int **extrema, ghost_lidx a, ghost_lidx b); 
ghost_error mat_bandwidth(ghost_sparsemat *mat, int *lower_bw, int *upper_bw, int a, int b);
ghost_error checker(ghost_sparsemat *mat);
ghost_error split_transition(ghost_sparsemat *mat);
//to be used if requirement is satisfied
ghost_error split_analytical(ghost_sparsemat *mat);


#endif

