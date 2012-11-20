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

static mat_data_t rhsVal (int i) 
{
#ifdef COMPLEX
	return i+1.0 + I*(i+1.5);
#else
	return i+1.0 ;
#endif
}

int main( int argc, char* argv[] ) 
{

	int  kernel, nIter = 1;
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
	GHOST_SPM_GPUFORMATS *matrixFormats = NULL;

	
	unsigned int sortBlock = 4;
	mat_trait_t trait = {.format = GHOST_SPMFORMAT_BJDS, 
		.flags = GHOST_SPM_SORTED | GHOST_SPM_PERMUTECOLIDX /*GHOST_SPM_DEFAULT*/,
		.aux = &sortBlock};
	mat_trait_t traits[3] = {trait,trait,trait};

	
	SpMVM_init(argc,argv,options);       // basic initialization
	setup = SpMVM_createSetup(matrixPath,traits,3,GHOST_SETUP_GLOBAL,matrixFormats);
	lhs   = SpMVM_createVector(setup,ghost_vec_t_LHS,NULL);
	rhs   = SpMVM_createVector(setup,ghost_vec_t_RHS,rhsVal);

#ifdef CHECK	
	ghost_vec_t *goldLHS = SpMVM_referenceSolver(matrixPath,setup,rhsVal,nIter,options);	
#endif

	SpMVM_printEnvInfo();
	SpMVM_printSetupInfo(setup,options);
	SpMVM_printHeader("Performance");

	for (kernel=0; kernel < nKernels; kernel++){

		time = SpMVM_solve(lhs,setup,rhs,kernels[kernel],nIter);

		if (time < 0.) {
			SpMVM_printLine(SpMVM_modeName(kernels[kernel]),NULL,"SKIPPED");
			continue;
		}
		
#ifdef CHECK
		errcount=0;
		for (i=0; i<setup->lnrows; i++){
			mytol = EPSILON * ABS(goldLHS->val[i]); 
			if (REAL(ABS(goldLHS->val[i]-lhs->val[i])) > mytol || 
					IMAG(ABS(goldLHS->val[i]-lhs->val[i])) > mytol){
				printf( "PE%d: error in row %"PRmatIDX": %.2e + %.2ei vs. %.2e +"
						"%.2ei (tol: %e, diff: %e)\n", SpMVM_getRank(), i, REAL(goldLHS->val[i]),
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
			SpMVM_printLine(SpMVM_modeName(kernels[kernel]),NULL,"FAILED");
		else
			SpMVM_printLine(SpMVM_modeName(kernels[kernel]),"GF/s","%f",
					FLOPS_PER_ENTRY*1.e-9*
					(double)setup->nnz/time,
					time*1.e3);
#else
		printf("%11s: %5.2f GF/s | %5.2f ms/it\n",
				SpMVM_modeName(kernels[kernel]),
				FLOPS_PER_ENTRY*1.e-9*
				(double)setup->nnz/time,
				time*1.e3);
#endif

		SpMVM_zeroVector(lhs);

	}
	SpMVM_printFooter();


	SpMVM_freeVector( lhs );
	SpMVM_freeVector( rhs );
	//	SpMVM_freeLCRP( lcrp );

#ifdef CHECK
	SpMVM_freeVector( goldLHS );
	//SpMVM_freeCRS( cr );
#endif

	SpMVM_finish();

	return EXIT_SUCCESS;

}
