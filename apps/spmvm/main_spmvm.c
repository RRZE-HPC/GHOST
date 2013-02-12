#include <ghost.h>
#include <ghost_util.h>

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

GHOST_REGISTER_DT_Z(vecdt)

static void rhsVal (int i, void *val) 
{
	UNUSED(i);
	*(vecdt_t *)val = 1+I*1;//i + (vecdt_t)1.0 + I*i;
}
int main( int argc, char* argv[] ) 
{

	int  mode, nIter = 1;
	double time;

#ifdef CHECK
	ghost_midx_t i, errcount = 0;
	double mytol;
#endif

	int ghostOptions = GHOST_OPTION_NONE; // TODO remote part immer axpy
	int modes[] = {GHOST_SPMVM_MODE_NOMPI,
		GHOST_SPMVM_MODE_VECTORMODE,
		GHOST_SPMVM_MODE_GOODFAITH/*,
		GHOST_SPMVM_MODE_TASKMODE*/};
	int nModes = sizeof(modes)/sizeof(int);

	int spmvmOptions = GHOST_SPMVM_AXPY;

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
		trait.datatype = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL;
	}
	ghost_mtraits_t traits[3];
	traits[0] = trait; traits[1] = trait; traits[2] = trait;

	ghost_vtraits_t lvtraits = {.flags = GHOST_VEC_LHS,.aux = NULL,.datatype = vecdt};
	ghost_vtraits_t rvtraits = {.flags = GHOST_VEC_RHS,.aux = NULL,.datatype = vecdt};

	ghost_init(argc,argv,ghostOptions);       // basic initialization
	context = ghost_createContext(matrixPath,traits,3,GHOST_CONTEXT_DEFAULT);
	lhs   = ghost_createVector(context,&lvtraits);
	rhs   = ghost_createVector(context,&rvtraits);

	rhs->fromFunc(rhs,rhsVal);
//	rhs->fromRand(rhs);

#ifdef CHECK	
	ghost_vec_t *goldLHS = ghost_referenceSolver(matrixPath,context,rhs,nIter,spmvmOptions);	
#endif
	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printOptionsInfo(ghostOptions);
	ghost_printContextInfo(context);

	ghost_printHeader("Performance");

	for (mode=0; mode < nModes; mode++){

		int argOptions = spmvmOptions | modes[mode];
		time = ghost_bench_spmvm(lhs,context,rhs,&argOptions,nIter);

		if (time < 0.) {
			ghost_printLine(ghost_modeName(modes[mode]),NULL,"SKIPPED");
			continue;
		}

#ifdef CHECK
		errcount=0;
		vecdt_t res,ref;
		for (i=0; i<context->lnrows(context); i++){
			goldLHS->entry(goldLHS,i,&ref);
			lhs->entry(lhs,i,&res);
		
			mytol = 1e-16 * context->fullMatrix->rowLen(context->fullMatrix,i);
//			printf("%f + %fi vs. %f + %fi\n",creal(ref),cimag(ref),creal(res),cimag(res));
			if (creal(cabs(ref-res)) > creal(mytol) ||
					cimag(cabs(ref-res)) > cimag(mytol)){
				printf( "PE%d: error in %s, row %"PRmatIDX": %.2e + %.2ei vs. %.2e +"
						"%.2ei (tol: %.2e + %.2ei, diff: %e)\n", ghost_getRank(),ghost_modeName(modes[mode]), i, creal(ref),
						cimag(ref),
						creal(res),
						cimag(res),
						creal(mytol),cimag(mytol),creal(cabs(ref-res)));
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
			ghost_printLine(ghost_modeName(modes[mode]),NULL,"FAILED");
		else
			ghost_printLine(ghost_modeName(modes[mode]),"GF/s","%.2f",
					FLOPS_PER_ENTRY*1.e-9*
					(double)context->gnnz(context)/time);
#else
		ghost_printLine(ghost_modeName(modes[mode]),"GF/s","%.2f",
				FLOPS_PER_ENTRY*1.e-9*
				(double)context->gnnz(context)/time);
#endif

		lhs->zero(lhs);

	}
	ghost_printFooter();

	lhs->destroy(lhs);
	rhs->destroy(rhs);
	ghost_freeContext( context );

#ifdef CHECK
	goldLHS->destroy(goldLHS);
#endif

	ghost_finish();

	return EXIT_SUCCESS;

}
