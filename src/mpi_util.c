#define _GNU_SOURCE

#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/mpi_util.h"
#include "ghost/util.h"
#include "ghost/constants.h"
#include "ghost/affinity.h"
#include "ghost/log.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <err.h>
#include <fcntl.h>
#include <errno.h>

#include <math.h>
#include <complex.h>
#include <dlfcn.h>



static ghost_mpi_comm_t ghost_node_comm = MPI_COMM_NULL;
static int ghost_node_rank = -1;

MPI_Datatype ghost_mpi_dataType(int datatype)
{
    if (datatype & GHOST_DT_FLOAT) {
        if (datatype & GHOST_DT_COMPLEX)
            return GHOST_MPI_DT_C;
        else
            return MPI_FLOAT;
    } else {
        if (datatype & GHOST_DT_COMPLEX)
            return GHOST_MPI_DT_Z;
        else
            return MPI_DOUBLE;
    }
}

MPI_Op ghost_mpi_op_sum(int datatype)
{
    if (datatype & GHOST_DT_FLOAT) {
        if (datatype & GHOST_DT_COMPLEX) {
            return GHOST_MPI_OP_SUM_C;
        } else {
            return MPI_SUM;
        }
    } else {
        if (datatype & GHOST_DT_COMPLEX) {
            return GHOST_MPI_OP_SUM_Z;
        } else {
            return MPI_SUM;
        }
    }

}

