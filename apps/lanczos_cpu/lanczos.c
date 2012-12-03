#include <ghost.h>
#include <ghost_util.h>
#include <ghost_vec.h>
#include "lanczos.h"
#include <omp.h>

#include <mpi.h>
#include <limits.h>
#include <getopt.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int converged(ghost_mdat_t evmin)
{
	static ghost_mdat_t oldevmin = -1e9;

	int converged = ABS(evmin-oldevmin) < 1e-9;
	oldevmin = evmin;

	return converged;
}

static void dotprod(ghost_vec_t *v1, ghost_vec_t *v2, ghost_mdat_t *res, int n)
{
	int i;
	ghost_mdat_t sum = 0;
#pragma omp parallel for private(i) reduction(+:sum)
		for (i=0; i<n; i++)
			sum += v1->val[i]*v2->val[i];
	*res = sum;
}

static void axpy(ghost_vec_t *v1, ghost_vec_t *v2, ghost_mdat_t s, int n)
{
	int i;
#pragma omp parallel for private(i)
		for (i=0; i<n; i++)
			v1->val[i] = v1->val[i] + s*v2->val[i];

}

static void vecscal(ghost_vec_t *vec, ghost_mdat_t s, int n)
{
	int i;
#pragma omp parallel for private(i)
		for (i=0; i<n; i++)
			vec->val[i] = s*vec->val[i];
}


static void lanczosStep(ghost_setup_t *setup, ghost_vec_t *vnew, ghost_vec_t *vold,
		ghost_mdat_t *alpha, ghost_mdat_t *beta)
{
	vecscal(vnew,-*beta,setup->lnrows);
	ghost_solve(vnew, setup, vold, GHOST_MODE_NOMPI, 1);
	dotprod(vnew,vold,alpha,setup->lnrows);
	axpy(vnew,vold,-(*alpha),setup->lnrows);
	dotprod(vnew,vnew,beta,setup->lnrows);
	*beta=SQRT(*beta);
	vecscal(vnew,1./(*beta),setup->lnrows);
}

static ghost_mdat_t rhsVal (int i)
{
	return i+1.0;
}

int main( int argc, char* argv[] )
{
	int ferr;
	ghost_mdat_t alpha=0., beta=0.;
	int n;

	ghost_setup_t *setup;
	ghost_vec_t   *vold;
	ghost_vec_t   *vnew;
	ghost_vec_t   *r0;

	int iteration, nIter = 500;
	char *matrixPath = argv[1];
	int options = GHOST_OPTION_AXPY;
	ghost_mtraits_t trait = {.format="CRS",
		.flags=GHOST_SPM_DEFAULT,
		.aux=NULL};

	ghost_init(argc,argv,options);       // basic initialization
	
	setup = ghost_createSetup(matrixPath,&trait,1,GHOST_SETUP_GLOBAL,NULL);
	vnew  = ghost_createVector(setup,GHOST_VEC_RHS|GHOST_VEC_LHS,NULL);
	r0    = ghost_createVector(setup,GHOST_VEC_GLOBAL,rhsVal); 
	
	ghost_normalizeVector(r0); // normalize the global vector r0

	vold = ghost_distributeVector(setup->communicator,r0); // scatter r0 to vold

	ghost_mdat_t *alphas  = (ghost_mdat_t *)malloc(sizeof(ghost_mdat_t)*nIter);
	ghost_mdat_t *betas   = (ghost_mdat_t *)malloc(sizeof(ghost_mdat_t)*nIter);
	ghost_mdat_t *falphas = (ghost_mdat_t *)malloc(sizeof(ghost_mdat_t)*nIter);
	ghost_mdat_t *fbetas  = (ghost_mdat_t *)malloc(sizeof(ghost_mdat_t)*nIter);
	
	betas[0] = beta;

	for(iteration = 0, n=1; 
			iteration < nIter && !converged(falphas[0]); 
			iteration++, n++) 
	{
		printf("\r");

		lanczosStep(setup,vnew,vold,&alpha,&beta);
		ghost_swapVectors(vnew,vold);

		alphas[iteration] = alpha;
		betas[iteration+1] = beta;
		memcpy(falphas,alphas,n*sizeof(ghost_mdat_t)); // alphas and betas will be destroyed in imtql
		memcpy(fbetas,betas,n*sizeof(ghost_mdat_t));

		imtql1_(&n,falphas,fbetas,&ferr);
		if(ferr != 0) printf("Error: the %d. ev could not be determined\n",ferr);
		printf("e: %f",	falphas[0]);
	}
	printf("\n");

	ghost_freeVector(r0);
	ghost_freeVector(vold);
	ghost_freeVector(vnew);
	ghost_freeSetup (setup);

	ghost_finish();

	return EXIT_SUCCESS;

}
