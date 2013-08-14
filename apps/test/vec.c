#include <ghost.h>
#include <ghost_util.h>

GHOST_REGISTER_DT_D(dbl)

int main()
{

	int nr = 4, nc = 4;
	int r,c;
	ghost_vec_t *v1, *v2, *v3, *v4;


	ghost_vtraits_t v1traits = GHOST_VTRAITS_INIT(.nrows=nr, .nvecs=nc, .datatype=dbl);

	v1 = ghost_createVector(&v1traits);
	v1->fromRand(v1,NULL);
	printf("===== v1: (should be random)\n");
	v1->print(v1);
	printf("\n");

	v2 = v1->clone(v1);
	printf("===== v2: (should be a clone of v1)\n");
	v2->print(v2);
	printf("\n");
	
	v2 = v1->extract(v1,2,2,1,1);
	printf("===== v2: (should be the inner part of v1)\n");
	v2->print(v2);
	printf("\n");
	
	v3 = v1->view(v1,2,2,1,1);
	printf("===== v3: (should be a view of the inner part of v1)\n");
	v3->print(v3);
	printf("\n");
	
	v1->zero(v1);
	printf("===== v1: (should be zero)\n");
	v1->print(v1);
	printf("\n");
	
	printf("===== v2: (should not have changed)\n");
	v2->print(v2);
	printf("\n");
	
	printf("===== v3: (should be zero because it is a view)\n");
	v3->print(v3);
	printf("\n");
	
	ghost_vtraits_t *v4traits = ghost_cloneVtraits(&v1traits);
	v4traits->nvecs -= 1;
	v4traits->nrows -= 1;
	v4 = ghost_createVector(v4traits);

	dbl_t *data = (dbl_t *)ghost_malloc(sizeof(dbl_t)*v4->traits->nrowspadded*nc);
	for (c=0; c<nc; c++) {
		for (r=0; r<nr; r++) {
			data[c*v4->traits->nrowspadded+r] = c+0.1*r;
		}
	}
	
	v4->viewPlain(v4,data,nr-1,nc-1,1,1,v4->traits->nrowspadded);
	printf("===== v4: (should be plain data w/ offset 1 in both directions)\n");
	v4->print(v4);
	printf("\n");
}
