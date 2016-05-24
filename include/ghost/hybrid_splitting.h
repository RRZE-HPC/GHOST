#ifndef HYPRID_SPLITTING_H
#define HYBRID_SPLITTING_H

#include "sparsemat.h"

ghost_error checker(ghost_sparsemat *mat);
ghost_error split_transition(ghost_sparsemat *mat);

#endif

