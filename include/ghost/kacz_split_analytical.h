#ifndef KACZ_SPLITTING_ANALYTICAL_H
#define KACZ_SPLITTING_ANALYTICAL_H

#include "sparsemat.h"
#include "util.h"
#include "locality.h"
#include "bincrs.h"
#include "omp.h"
#include <omp.h>
#include <math.h>

ghost_error split_analytical(ghost_sparsemat *mat);

#endif

