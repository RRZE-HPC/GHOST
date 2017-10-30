#include "ghost/context.h"
#include "ghost/bench.h"
#include "ghost/bincrs.h"
#include "ghost/config.h"
#include "ghost/core.h"
#include "ghost/locality.h"
#include "ghost/log.h"
#include "ghost/machine.h"
#include "ghost/map.h"
#include "ghost/matrixmarket.h"
#include "ghost/omp.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include <float.h>
#include <iomanip>
#include <math.h>
#include <sstream>
#include <vector>
using namespace std;

ghost_error ghost_context_comm_string(char **str, ghost_context *ctx,
                                      int root) {
#ifdef GHOST_HAVE_MPI
  stringstream strbuf;

  int nrank, r, p, me, l;
  bool printline = false;
  ghost_lidx maxdues = 0, maxwishes = 0;
  int dueslen = 0, duesprintlen = 4, wisheslen = 0, wishesprintlen = 6, ranklen,
      rankprintlen, linelen;
  ghost_nrank(&nrank, ctx->mpicomm);
  ghost_rank(&me, ctx->mpicomm);

  ranklen = (int)floor(log10(abs(nrank))) + 1;

  for (r = 0; r < nrank; r++) {
    maxdues = MAX(maxdues, ctx->dues[r]);
    maxwishes = MAX(maxwishes, ctx->wishes[r]);
  }

  MPI_Allreduce(MPI_IN_PLACE, &maxdues, 1, ghost_mpi_dt_lidx, MPI_MAX,
                ctx->mpicomm);
  MPI_Allreduce(MPI_IN_PLACE, &maxwishes, 1, ghost_mpi_dt_lidx, MPI_MAX,
                ctx->mpicomm);

  if (maxdues != 0) {
    dueslen = (int)floor(log10(abs(maxdues))) + 1;
  }

  if (maxwishes != 0) {
    wisheslen = (int)floor(log10(abs(maxwishes))) + 1;
  }

  rankprintlen = MAX(4, ranklen);
  duesprintlen = MAX(4, dueslen + 3 + ranklen);
  wishesprintlen = MAX(6, wisheslen + 3 + ranklen);

  linelen = rankprintlen + duesprintlen + wishesprintlen + 4;

  if (me == root) {
    for (l = 0; l < linelen; l++)
      strbuf << "=";
    strbuf << "\n";
    strbuf << setw(rankprintlen) << "RANK"
           << "  " << setw(duesprintlen) << "DUES"
           << "  " << setw(wishesprintlen) << "WISHES\n";
    for (l = 0; l < linelen; l++)
      strbuf << "=";
    strbuf << "\n";
  }

  vector<ghost_lidx> dues(nrank);
  vector<ghost_lidx> wishes(nrank);

  for (r = 0; r < nrank; r++) {
    if (r == root && me == root) {
      memcpy(wishes.data(), ctx->wishes, nrank * sizeof(ghost_lidx));
      memcpy(dues.data(), ctx->dues, nrank * sizeof(ghost_lidx));
    } else {
      if (me == root) {
        MPI_Recv(wishes.data(), nrank, ghost_mpi_dt_lidx, r, r, ctx->mpicomm,
                 MPI_STATUS_IGNORE);
        MPI_Recv(dues.data(), nrank, ghost_mpi_dt_lidx, r, nrank + r,
                 ctx->mpicomm, MPI_STATUS_IGNORE);
      }
      if (me == r) {
        MPI_Send(ctx->wishes, nrank, ghost_mpi_dt_lidx, root, me, ctx->mpicomm);
        MPI_Send(ctx->dues, nrank, ghost_mpi_dt_lidx, root, me + nrank,
                 ctx->mpicomm);
      }
    }

    if (me == root) {
      for (p = 0; p < nrank; p++) {
        if (wishes[p] && dues[p]) {
          strbuf << setw(rankprintlen) << r;
          strbuf << "  =>" << setw(ranklen) << p << " " << setw(dueslen)
                 << dues[p];
          strbuf << setw(MAX(2, 2 + wishesprintlen - (wisheslen + 3 + ranklen)))
                 << " "
                 << "<=" << setw(ranklen) << p << " " << setw(wisheslen)
                 << wishes[p] << "\n";

        } else if (wishes[p]) {
          strbuf << setw(rankprintlen) << r << "  " << setw(duesprintlen)
                 << " ";
          strbuf << setw(MAX(2, 2 + wishesprintlen - (wisheslen + 3 + ranklen)))
                 << " "
                 << "<=" << setw(ranklen) << p << " " << setw(wisheslen)
                 << wishes[p] << "\n";
        } else if (dues[p]) {
          strbuf << setw(rankprintlen) << r << "  =>" << setw(ranklen) << p
                 << " " << setw(dueslen) << dues[p] << "\n";
        }
        if (wishes[p] || dues[p]) {
          printline = true;
        }
      }
      if (printline && r != nrank - 1) {
        for (l = 0; l < linelen; l++)
          strbuf << "-";
        strbuf << "\n";
      }
    }
    printline = false;
  }

  if (me == root) {
    for (l = 0; l < linelen; l++)
      strbuf << "=";
    *str = (char *)malloc(strbuf.str().size());
    if (*str != NULL)
      strncpy(*str, strbuf.str().c_str(), strbuf.str().size());
  }
#else
  UNUSED(ctx);
  UNUSED(root);
  ghost_malloc((void **)str, 8);
  strcpy(*str, "No MPI!");
#endif

  return GHOST_SUCCESS;
}
