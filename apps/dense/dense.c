#include <stdio.h>
#include <ghost.h>
#include <ghost_util.h>
#include <ghost_vec.h>

#include <unistd.h>

GHOST_REGISTER_DT_D(vecdt)
GHOST_REGISTER_DT_D(matdt)

int main(int argc, char* argv[]) 
{
	int nv = 2;
	vecdt_t one = 1.;
	vecdt_t dotpr[nv];

	ghost_vtraits_t dmtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS, .nvecs=nv, .datatype=vecdt);

	ghost_context_t *ctx;
	ghost_vec_t *dm1, *dm2, *dm3;

	ghost_init(argc,argv);

	ctx = ghost_createContext(8,8,GHOST_CONTEXT_DEFAULT,NULL,MPI_COMM_WORLD);
	dm1 = ghost_createVector(ctx,&dmtraits);
	dm2 = ghost_createVector(ctx,&dmtraits);

	dm1->fromScalar(dm1,&one);
	dm2->fromScalar(dm2,&one);

	dm1->print(dm1);

	dm2->print(dm2);

	for (int i=0; i<nv; i++)
		dotpr[i] = 0.;

	ghost_dotProduct(dm1,dm2,&dotpr);
	printf("dotproduct:\n");
	for (int i=0; i<nv; i++)
		printf("%f ",dotpr[i]);
	printf("\n");

	ghost_vtraits_t dm3traits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_DEFAULT, .nrows = dm1->traits->nvecs, .nvecs = dm2->traits->nvecs, .datatype=vecdt);
	dm3 = ghost_createVector(ctx,&dm3traits);
	dm3->fromScalar(dm3,&one);

	vecdt_t alpha = 1., beta = 1.;
	ghost_gemm("T",dm1,dm2,dm3,&alpha,&beta,GHOST_GEMM_ALL_REDUCE);

	dm3->print(dm3);

	printf("\n##### write\n");
	dm2->print(dm2);
	printf("##### read\n");
	ghost_vecToFile(dm2,"foo.vec");
	ghost_vecFromFile(dm2,"foo.vec");
	dm2->print(dm2);
	printf("#####\n");

	dm1->destroy(dm1);
	dm2->destroy(dm2);
	dm3->destroy(dm3);

	ghost_finish();

	return EXIT_SUCCESS;
}
