#include <ghost.h>
#include <ghost_util.h>
#include <ghost_vec.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <libgen.h>
#ifdef MPI
#include <mpi.h>
#endif

#define CHECK // compare with reference solution

extern int optind;

static ghost_mdat_t rhsVal (int i) 
{
#ifdef COMPLEX
	return i+1.0 + I*(i+1.5);
#else
	return i+1.0 ;
#endif
}

int main( int argc, char* argv[] ) 
{

	int  kernel, nIter = 100;
	double time;

#ifdef CHECK
	mat_idx_t i, errcount = 0;
	double mytol;
#endif

	int options = GHOST_OPTION_AXPY; // TODO remote kernel immer axpy
	int kernels[] = {GHOST_MODE_NOMPI,
		GHOST_MODE_VECTORMODE,
		GHOST_MODE_GOODFAITH,
		GHOST_MODE_TASKMODE};
	int nKernels = sizeof(kernels)/sizeof(int);


	ghost_vec_t *lhs; // lhs vector
	ghost_vec_t *rhs; // rhs vector

	ghost_setup_t *setup;

	char *matrixPath = argv[1];

	ghost_mtraits_t trait;
	if (argc == 5) {
		trait.format = argv[2];
		trait.flags = atoi(argv[3]);
		unsigned int sortBlock = (unsigned int)atoi(argv[4]);
		trait.aux = &sortBlock;

	} else {
		trait.format = "CRS";
		trait.flags = GHOST_SPM_DEFAULT;
		trait.aux = NULL;
	}
	ghost_mtraits_t traits[3] = {trait,trait,trait};


	ghost_init(argc,argv,options);       // basic initialization
	setup = ghost_createSetup(matrixPath,traits,3,GHOST_SETUP_GLOBAL);
	lhs   = ghost_createVector(setup,GHOST_VEC_LHS,NULL);
	rhs   = ghost_createVector(setup,GHOST_VEC_RHS,rhsVal);

#ifdef CHECK	
	ghost_vec_t *goldLHS = ghost_referenceSolver(matrixPath,setup,rhsVal,nIter,options);	
#endif

	ghost_printEnvInfo();
	ghost_printSetupInfo(setup,options);
	ghost_printHeader("Performance");

	for (kernel=0; kernel < nKernels; kernel++){

		time = ghost_solve(lhs,setup,rhs,kernels[kernel],nIter);

		if (time < 0.) {
			ghost_printLine(ghost_modeName(kernels[kernel]),NULL,"SKIPPED");
			continue;
		}

#ifdef CHECK
		errcount=0;
		for (i=0; i<setup->lnrows; i++){
			mytol = EPSILON * ABS(goldLHS->val[i]); 
			if (REAL(ABS(goldLHS->val[i]-lhs->val[i])) > mytol || 
					IMAG(ABS(goldLHS->val[i]-lhs->val[i])) > mytol){
				printf( "PE%d: error in row %"PRmatIDX": %.2e + %.2ei vs. %.2e +"
						"%.2ei (tol: %e, diff: %e)\n", ghost_getRank(), i, REAL(goldLHS->val[i]),
						IMAG(goldLHS->val[i]),
						REAL(lhs->val[i]),
						IMAG(lhs->val[i]),
						mytol,REAL(ABS(goldLHS->val[i]-lhs->val[i])));
				errcount++;
			}
		}
		mat_idx_t totalerrors;
#ifdef MPI
		MPI_safecall(MPI_Allreduce(&errcount,&totalerrors,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD));
#else
		totalerrors = errcount;
#endif
		if (totalerrors)
			ghost_printLine(ghost_modeName(kernels[kernel]),NULL,"FAILED");
		else
			ghost_printLine(ghost_modeName(kernels[kernel]),"GF/s","%f",
					FLOPS_PER_ENTRY*1.e-9*
					(double)setup->nnz/time);
#else
		ghost_printLine(ghost_modeName(kernels[kernel]),"GF/s","%f",
				FLOPS_PER_ENTRY*1.e-9*
				(double)setup->nnz/time);
#endif

		ghost_zeroVector(lhs);

	}
	ghost_printFooter();


	ghost_freeVector( lhs );
	ghost_freeVector( rhs );
	ghost_freeSetup( setup );

#ifdef CHECK
	ghost_freeVector( goldLHS );
#endif

	ghost_finish();

	return EXIT_SUCCESS;

}
