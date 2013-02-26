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

GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

static void rhsVal (int i, int v, void *val) 
{
	UNUSED(i);
	UNUSED(v);
	*(vecdt_t *)val = 1+I*1;//i + (vecdt_t)1.0 + I*i;
}

int main( int argc, char* argv[] ) 
{

	int  mode, nIter = 100;
	double time;
	vecdt_t zero = 0.;

#ifdef CHECK
	ghost_midx_t i, errcount = 0;
	double mytol;
#endif

	int modes[] = {//GHOST_SPMVM_MODE_NOMPI,
		GHOST_SPMVM_MODE_VECTORMODE,
		GHOST_SPMVM_MODE_GOODFAITH/*,
		GHOST_SPMVM_MODE_TASKMODE*/};
	int nModes = sizeof(modes)/sizeof(int);

	int spmvmOptions = GHOST_SPMVM_AXPY;
	ghost_matfile_header_t fileheader;

	ghost_mat_t *mat; // matrix
	ghost_vec_t *lhs; // lhs vector
	ghost_vec_t *rhs; // rhs vector

	ghost_context_t *context;

	char *matrixPath = argv[1];
	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = "ELLPACK", .datatype = matdt);
	ghost_vtraits_t lvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .datatype = vecdt);
	ghost_vtraits_t rvtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS, .datatype = vecdt);

	if (argc == 5) {
		mtraits.format = argv[2];
		mtraits.flags = atoi(argv[3]);
		int sortBlock = atoi(argv[4]);
		mtraits.aux = &sortBlock;
	}

	ghost_init(argc,argv);       // basic initialization
	ghost_pinThreads(GHOST_PIN_PHYS,NULL);
	
	ghost_readMatFileHeader(matrixPath,&fileheader);
	context = ghost_createContext(fileheader.nrows,GHOST_CONTEXT_DEFAULT);
	mat = ghost_createMatrix(&mtraits,1);
	lhs = ghost_createVector(&lvtraits);
	rhs = ghost_createVector(&rvtraits);

	mat->fromFile(mat,context,matrixPath);
	lhs->fromScalar(lhs,context,&zero);
	rhs->fromFunc(rhs,context,rhsVal);

#ifdef CHECK	
	ghost_vec_t *goldLHS = ghost_referenceSolver(matrixPath,matdt,context,rhs,nIter,spmvmOptions);	
#endif
//	ghost_printSysInfo();
	ghost_printGhostInfo();
	ghost_printContextInfo(context);
	ghost_printMatrixInfo(mat);

	ghost_printHeader("Performance");

	for (mode=0; mode < nModes; mode++){

		int argOptions = spmvmOptions | modes[mode];
		time = ghost_bench_spmvm(context,lhs,mat,rhs,&argOptions,nIter);

		if (time < 0.) {
			ghost_printLine(ghost_modeName(modes[mode]),NULL,"SKIPPED");
			continue;
		}

#ifdef CHECK
		errcount=0;
		vecdt_t res,ref;
		for (i=0; i<mat->nrows(mat); i++){
			goldLHS->entry(goldLHS,i,&ref);
			lhs->entry(lhs,i,&res);
		
			mytol = 1e-8 * mat->rowLen(mat,i);
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
					(double)ghost_getMatNnz(mat)/time);
#else
		ghost_printLine(ghost_modeName(modes[mode]),"GF/s","%.2f",
				FLOPS_PER_ENTRY*1.e-9*
				(double)ghost_getMatNnz(mat)/time);
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
