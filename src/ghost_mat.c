#define _XOPEN_SOURCE 600
#include "ghost_mat.h"
#include "ghost.h"
#include "ghost_util.h"

#include <string.h>
#include <libgen.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>





int isMMfile(const char *filename) 
{

	FILE *file = fopen( filename, "r" );

	if( ! file ) {
		ABORT("Could not open file in isMMfile: %s",filename);
	}

	const char *keyword="%%MatrixMarket";
	char *readkw = (char *)ghost_malloc((strlen(keyword)+1)*sizeof(char));
	if (NULL == fgets(readkw,strlen(keyword)+1,file))
		return 0;

	int cmp = strcmp(readkw,keyword);

	free(readkw);
	fclose(file);
	return cmp==0?1:0;
}

/*ghost_mm_t * readMMFile(const char* filename ) 
{
	MM_typecode matcode;
	FILE *f;
	ghost_midx_t i;
	ghost_mm_t* mm = (ghost_mm_t*) malloc( sizeof( ghost_mm_t ) );

	if ((f = fopen(filename, "r")) == NULL) 
		exit(1);

	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

#ifdef GHOST_MAT_COMPLEX
	if (!mm_is_complex(matcode))
		DEBUG_LOG(0,"Warning! The library has been built for complex data "
				"but the MM file contains real data. Casting...");
#else
	if (mm_is_complex(matcode))
		DEBUG_LOG(0,"Warning! The library has been built for real data "
				"but the MM file contains complex data. Casting...");
#endif



	if ((mm_read_mtx_crd_size(f, &mm->nrows, &mm->ncols, &mm->nEnts)) !=0)
		exit(1);


	mm->nze = (NZE_TYPE *)malloc(mm->nEnts*sizeof(NZE_TYPE));

	if (!mm_is_complex(matcode)) {
		for (i=0; i<mm->nEnts; i++)
		{
#ifdef GHOST_MAT_DP
			double re;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %lg\n", &mm->nze[i].row, &mm->nze[i].col, &re);
#else
			float re;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %g\n", &mm->nze[i].row, &mm->nze[i].col, &re);
#endif
#ifdef GHOST_MAT_COMPLEX
			mm->nze[i].val = re+I*0;
#else
			mm->nze[i].val = re;
#endif
			mm->nze[i].col--;  // adjust from 1-based to 0-based 
			mm->nze[i].row--;
		}
	} else {

		for (i=0; i<mm->nEnts; i++)
		{
#ifdef GHOST_MAT_DP
			double re,im;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %lg %lg\n", &mm->nze[i].row, &mm->nze[i].col, &re,
					&im);
#else
			float re,im;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %g %g\n", &mm->nze[i].row, &mm->nze[i].col, &re,
					&im);
#endif
#ifdef GHOST_MAT_COMPLEX	
			mm->nze[i].val = re+I*im;
#else
			mm->nze[i].val = re;
#endif
			mm->nze[i].col--; //  adjust from 1-based to 0-based
			mm->nze[i].row--;
		}

	}


	if (f !=stdin) fclose(f);
	return mm;
}*/
