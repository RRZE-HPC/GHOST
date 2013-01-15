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

//#define CHECK // compare with reference solution

extern int optind;

static ghost_vdat_t rhsVal (int i) 
{
#ifdef GHOST_VEC_COMPLEX
	return (ghost_vdat_el_t)(i+1.0) + I*(ghost_vdat_el_t)(i+1.5);
#else
	return i+1.0 ;
#endif
}

int main( int argc, char* argv[] ) 
{

	int  kernel, nIter = 50;
	double time;

#ifdef CHECK
	ghost_midx_t i, errcount = 0;
	ghost_vdat_t mytol;
#endif

	int options = GHOST_SPMVM_AXPY; // TODO remote kernel immer axpy
	int kernels[] = {GHOST_MODE_NOMPI,
		GHOST_MODE_VECTORMODE,
		GHOST_MODE_GOODFAITH,
		GHOST_MODE_TASKMODE};
	int nKernels = sizeof(kernels)/sizeof(int);


	ghost_vec_t *lhs; // lhs vector
	ghost_vec_t *rhs; // rhs vector

	ghost_context_t *context;

	char *matrixPath = argv[1];

	ghost_mtraits_t trait;
	if (argc == 5) {
		trait.format = argv[2];
		trait.flags = atoi(argv[3]);
		unsigned int sortBlock = (unsigned int)atoi(argv[4]);
		trait.aux = &sortBlock;

	} else {
//		unsigned int aux = 256;
		trait.format = "CRS";
		trait.flags = GHOST_SPM_DEFAULT;//GHOST_SPM_SORTED|GHOST_SPM_PERMUTECOLIDX;
		trait.aux = NULL;//&aux;
	}
	ghost_mtraits_t traits[3] = {trait,trait,trait};


	ghost_init(argc,argv,options);       // basic initialization
	context = ghost_createContext(matrixPath,traits,3,GHOST_CONTEXT_DEFAULT);
	lhs   = ghost_createVector(context,GHOST_VEC_LHS,NULL);
	rhs   = ghost_createVector(context,GHOST_VEC_RHS,rhsVal);

#ifdef CHECK	
	ghost_vec_t *goldLHS = ghost_referenceSolver(matrixPath,context,rhsVal,nIter,options);	
#endif

	ghost_printEnvInfo();
	ghost_printContextInfo(context,options);
	ghost_printHeader("Performance");

	for (kernel=0; kernel < nKernels; kernel++){

		time = ghost_spmvm(lhs,context,rhs,kernels[kernel],nIter);

		if (time < 0.) {
			ghost_printLine(ghost_modeName(kernels[kernel]),NULL,"SKIPPED");
			continue;
		}

#ifdef CHECK
		errcount=0;
		for (i=0; i<context->lnrows(context); i++){
			mytol = EPSILON * VABS(goldLHS->val[i]); 
			if (VREAL(VABS(goldLHS->val[i]-lhs->val[i])) > VREAL(mytol) ||
					VIMAG(VABS(goldLHS->val[i]-lhs->val[i])) > VIMAG(mytol)){
				printf( "PE%d: error in row %"PRmatIDX": %.2e + %.2ei vs. %.2e +"
						"%.2ei (tol: %.2e + %.2ei, diff: %e)\n", ghost_getRank(), i, VREAL(goldLHS->val[i]),
						VIMAG(goldLHS->val[i]),
						VREAL(lhs->val[i]),
						VIMAG(lhs->val[i]),
						VREAL(mytol),VIMAG(mytol),VREAL(VABS(goldLHS->val[i]-lhs->val[i])));
				errcount++;
			}
		}
		ghost_midx_t totalerrors;
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
					(double)context->gnnz(context)/time);
#else
		ghost_printLine(ghost_modeName(kernels[kernel]),"GF/s","%f",
				FLOPS_PER_ENTRY*1.e-9*
				(double)context->gnnz(context)/time);
#endif

		ghost_zeroVector(lhs);

	}
	ghost_printFooter();


	ghost_freeVector( lhs );
	ghost_freeVector( rhs );
	ghost_freeContext( context );

#ifdef CHECK
	ghost_freeVector( goldLHS );
#endif

	ghost_finish();

	return EXIT_SUCCESS;

}
