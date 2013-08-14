#include <stdio.h>
#include <ghost.h>
#include <ghost_util.h>

#include <omp.h>
#include <unistd.h>

GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

int main(int argc, char* argv[]) 
{
	vecdt_t zero = 0.;
	vecdt_t one = 1.;

	ghost_mtraits_t mtraits = GHOST_MTRAITS_INIT(.format = GHOST_SPM_FORMAT_CRS,.datatype = matdt);
	ghost_vtraits_t dmtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .nvecs=2, .datatype=vecdt);

	ghost_matfile_header_t fileheader;
	ghost_context_t *ctx;
	ghost_vec_t *dm1, *dm2, *dm3;
	ghost_mat_t *mat;

	ghost_init(argc,argv);

	ghost_readMatFileHeader(argv[1],&fileheader);

	ctx = ghost_createContext(fileheader.nrows,fileheader.ncols,GHOST_CONTEXT_DEFAULT);
	mat = ghost_createMatrix(&mtraits,1); // FIXME we need the matrix in order to have the distribution. fix this bug!
	dm1 = ghost_createVector(&dmtraits);
	dm2 = ghost_createVector(&dmtraits);

	mat->fromFile(mat,ctx,argv[1]);
	dm1->fromScalar(dm1,ctx,&one);
	dm2->fromScalar(dm2,ctx,&one);
	
	dm1->print(dm1);

	dm2->print(dm2);

	ghost_vtraits_t dm3traits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_DEFAULT, .nrows = dm1->traits->nvecs, .nvecs = dm2->traits->nvecs, .datatype=vecdt);
	dm3 = ghost_createVector(&dm3traits);
	dm3->fromScalar(dm3,NULL,&one);

	vecdt_t alpha = 1., beta = 1.;
	ghost_gemm("T",dm1,dm2,dm3,&alpha,&beta,GHOST_GEMM_ALL_REDUCE);

	dm3->print(dm3);

	printf("\n##### write\n");
	dm2->print(dm2);
	printf("##### read\n");
	ghost_vecToFile(dm2,"foo.vec",ctx);
	ghost_vecFromFile(dm2,"foo.vec",ctx);
	dm2->print(dm2);
	printf("#####\n");

	dm1->destroy(dm1);
	dm2->destroy(dm2);
	dm3->destroy(dm3);

	ghost_finish();

	return EXIT_SUCCESS;
}
