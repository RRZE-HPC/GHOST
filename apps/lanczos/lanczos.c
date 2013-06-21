#include <ghost.h>
#include <ghost_util.h>
#include <stdio.h>
#include <string.h>

GHOST_REGISTER_DT_D(vecdt); // vectors have double values
GHOST_REGISTER_DT_D(matdt); // matrix has double values

extern void imtql1_(int *, matdt_t *, matdt_t *, int *);
extern void imtql1f_(int *, matdt_t *, matdt_t *, int *);

static int converged(matdt_t evmin)
{
	static matdt_t oldevmin = -1e9;

	int converged = fabs(evmin-oldevmin) < 1e-9;
	oldevmin = evmin;

	return converged;
}

static void lanczosStep(ghost_context_t *context, ghost_mat_t *mat, ghost_vec_t *vnew, ghost_vec_t *vold,
		matdt_t *alpha, matdt_t *beta)
{
	int opt = GHOST_SPMVM_AXPY;
	matdt_t minusbeta = -*beta;
	vnew->scale(vnew,&minusbeta);
	ghost_spmvm(context, vnew, mat, vold, &opt);
	ghost_dotProduct(vnew,vold,alpha);
	matdt_t minusalpha = -*alpha;
	vnew->axpy(vnew,vold,&minusalpha);
	ghost_dotProduct(vnew,vnew,beta);
	*beta=sqrt(*beta);
	matdt_t recbeta = 1./(*beta);
	vnew->scale(vnew,&recbeta);
}

int main( int argc, char* argv[] )
{
	matdt_t alpha=0., beta=0., minusbeta, recbeta, minusalpha;
	int ferr, n, iteration, nIter = 500;
	char *matrixPath = argv[1];
	double zero = 0.;

	ghost_context_t *context;
	ghost_mat_t *mat;
	ghost_vec_t *vold;
	ghost_vec_t *vnew;
	ghost_matfile_header_t fileheader;
	
	ghost_init(argc,argv); // has to be the first call
	ghost_pinThreads(GHOST_PIN_PHYS,NULL); // pin the threads to the physical cores (no SMT)
	
	ghost_readMatFileHeader(matrixPath,&fileheader); // read basic matrix information
	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = GHOST_SPM_FORMAT_CRS, .datatype = matdt);
	ghost_vtraits_t vtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS|GHOST_VEC_RHS,.datatype = vecdt);

	context = ghost_createContext(fileheader.nrows,GHOST_CONTEXT_DEFAULT);
	mat   = ghost_createMatrix(&mtraits,1);
	vnew  = ghost_createVector(&vtraits);
	vold  = ghost_createVector(&vtraits);

	mat->fromFile(mat,context,matrixPath);
	vnew->fromScalar(vnew,context,&zero); // vnew = 0
	vold->fromRand(vold,context); // vold = random
	ghost_normalizeVec(vold); // normalize vold
	
	matdt_t *alphas  = (matdt_t *)ghost_malloc(sizeof(matdt_t)*nIter);
	matdt_t *betas   = (matdt_t *)ghost_malloc(sizeof(matdt_t)*nIter);
	matdt_t *falphas = (matdt_t *)ghost_malloc(sizeof(matdt_t)*nIter);
	matdt_t *fbetas  = (matdt_t *)ghost_malloc(sizeof(matdt_t)*nIter);

	betas[0] = beta;

	for(iteration = 0, n=1; 
			iteration < nIter && !converged(falphas[0]); 
			iteration++, n++) 
	{
		printf("\r");

		lanczosStep(context,mat,vnew,vold,&alpha,&beta);
		vnew->swap(vnew,vold);

		alphas[iteration] = alpha;
		betas[iteration+1] = beta;
		memcpy(falphas,alphas,n*sizeof(matdt_t)); // alphas and betas will be destroyed in imtql
		memcpy(fbetas,betas,n*sizeof(matdt_t));

		imtql1_(&n,falphas,fbetas,&ferr);

		if(ferr != 0) printf("Error: the %d. eigenvalue could not be determined\n",ferr);
		if (ghost_getRank() == 0)
			printf("minimal eigenvalue: %f", falphas[0]);
		fflush(stdout);
	}
	if (ghost_getRank() == 0)
		printf("%s\n",converged(falphas[0])?" (converged!)":" (max. iterations reached!)");

	vold->destroy(vold);
	vnew->destroy(vnew);
	ghost_freeContext(context);
	mat->destroy(mat);

	ghost_finish();
	
	return EXIT_SUCCESS;
}
