#include <spmvm.h>
#include <stdio.h>
#include <stdlib.h>
#include <mmio.h>


typedef struct {
	int row, col, nThEntryInRow;
	double real, imag ;
} NZE_TYPE;
typedef struct {
	int nRows, nCols, nEnts;
	NZE_TYPE* nze;
} MM_TYPE;
typedef struct {
	int nRows, nCols, nEnts;
	int* rowOffset;
	int* col;
	double *real, *imag;
} MY_CR_TYPE;

static int compareNZEPos( const void* a, const void* b ) {

	int aRow = ((NZE_TYPE*)a)->row,
		bRow = ((NZE_TYPE*)b)->row,
		aCol = ((NZE_TYPE*)a)->col,
		bCol = ((NZE_TYPE*)b)->col;

	if( aRow == bRow ) {
		return aCol - bCol;
	}
	else return aRow - bRow;
}
static void writeCR(MY_CR_TYPE* cr, char* filename, int datatype){

	FILE *file;
	int i;

	if ((file = fopen(filename, "wb"))==NULL){
		printf("Fehler beim Oeffnen von %s\n", filename);
		exit(1);
	}

	fwrite(&datatype,				sizeof(int),	1,			file);
	fwrite(&cr->nRows,               sizeof(int),    1,           file);
	fwrite(&cr->nCols,               sizeof(int),    1,           file);
	fwrite(&cr->nEnts,               sizeof(int),    1,           file);
	fwrite(&cr->rowOffset[0],        sizeof(int),    cr->nRows+1, file);
	fwrite(&cr->col[0],              sizeof(int),    cr->nEnts,   file);

	switch (datatype) {
		case DATATYPE_FLOAT:
			{
				float val;
				for (i=0; i<cr->nEnts; i++) {
					val = (float)cr->real[i];
					fwrite(&val, sizeof(float), 1, file);
				}
				break;
			}
		case DATATYPE_DOUBLE:
			{
				fwrite(&cr->real[0], sizeof(double), cr->nEnts, file);
				break;
			}
		case DATATYPE_COMPLEX_FLOAT:
			{
				float real;
				float imag;
				for (i=0; i<cr->nEnts; i++) {
					real = (float)cr->real[i];
					imag = (float)cr->imag[i];
					fwrite(&real,sizeof(float), 1,file);
					fwrite(&imag,sizeof(float), 1,file);
				}
				break;
			}
		case DATATYPE_COMPLEX_DOUBLE:
			{
				for (i=0; i<cr->nEnts; i++) {
					fwrite(&cr->real[i],sizeof(double), 1,file);
					fwrite(&cr->imag[i],sizeof(double), 1,file);
				}
				break;
			}
	}


	fflush(file);
	fclose(file);

	return;
}

static MM_TYPE * readMMfile(char* filename ) {

	MM_typecode matcode;
	FILE *f;
	int i;
	MM_TYPE* mm = (MM_TYPE*) malloc( sizeof( MM_TYPE ) );

	if ((f = fopen(filename, "r")) == NULL) 
		exit(1);

	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}


	if ((mm_read_mtx_crd_size(f, &mm->nRows, &mm->nCols, &mm->nEnts)) !=0)
		exit(1);


	mm->nze = (NZE_TYPE *)malloc(mm->nEnts*sizeof(NZE_TYPE));

	if (!mm_is_complex(matcode)) {
		for (i=0; i<mm->nEnts; i++)
		{
			fscanf(f, "%d %d %lg\n", &mm->nze[i].row, &mm->nze[i].col, &mm->nze[i].real);
			mm->nze[i].col--;  /* adjust from 1-based to 0-based */
			mm->nze[i].row--;
		}
	} else {
		for (i=0; i<mm->nEnts; i++)
		{
			fscanf(f, "%d %d %lg %lg\n",  &mm->nze[i].row, &mm->nze[i].col, &mm->nze[i].real, &mm->nze[i].imag);
			mm->nze[i].col--;  /* adjust from 1-based to 0-based */
			mm->nze[i].row--;
		}
	}


	if (f !=stdin) fclose(f);
	return mm;
}

static MY_CR_TYPE* convertMMtoCRmatrix( MM_TYPE* mm ) {

	int* nEntsInRow;
	int i, e, pos;

	MY_CR_TYPE* cr   = (MY_CR_TYPE*) malloc( sizeof( MY_CR_TYPE ));
	cr->rowOffset = (int*)     malloc((mm->nRows+1) * sizeof(int));
	cr->col       = (int*)     malloc(mm->nEnts * sizeof(int));
	cr->real       = (double*)  malloc(mm->nEnts * sizeof(double));
	cr->imag       = (double*)  malloc(mm->nEnts * sizeof(double));
	nEntsInRow    = (int*)     malloc(mm->nRows * sizeof(int));

	cr->nRows = mm->nRows;
	cr->nCols = mm->nCols;
	cr->nEnts = mm->nEnts;
	for( i = 0; i < mm->nRows; i++ ) nEntsInRow[i] = 0;

	qsort( mm->nze, (size_t)(mm->nEnts), sizeof( NZE_TYPE ), compareNZEPos );

	for( e = 0; e < mm->nEnts; e++ ) nEntsInRow[mm->nze[e].row]++;

	pos = 0;
	cr->rowOffset[0] = pos;

	for( i = 0; i < mm->nRows; i++ ) {
		cr->rowOffset[i] = pos;
		pos += nEntsInRow[i];
	}
	cr->rowOffset[mm->nRows] = pos;

	for( i = 0; i < mm->nRows; i++ ) nEntsInRow[i] = 0;

	for(i=0; i<cr->nRows; ++i) {
		int start = cr->rowOffset[i];
		int end = cr->rowOffset[i+1];
		int j;
		for(j=start; j<end; j++) {
			cr->real[j] = 0.0;
			cr->imag[j] = 0.0;
			cr->col[j] = 0;
		}
	}
	for( e = 0; e < mm->nEnts; e++ ) {
		const int row = mm->nze[e].row,
			  col = mm->nze[e].col;
		const double real = mm->nze[e].real;
		const double imag = mm->nze[e].imag;
		pos = cr->rowOffset[row] + nEntsInRow[row];
		cr->col[pos] = col;
		cr->real[pos] = real;
		cr->imag[pos] = imag;

		nEntsInRow[row]++;
	}
	free( nEntsInRow );

	return cr;
}

int main(int argc, char **argv) {

	if (argc != 4) {
		fprintf(stderr,"Usage: MMtoCRS <MMfile> <CRSfile> <CRSformat>\n");
		fprintf(stderr,"\tformat 0 = float\n");
		fprintf(stderr,"\tformat 1 = double\n");
		fprintf(stderr,"\tformat 2 = complex float\n");
		fprintf(stderr,"\tformat 3 = complex double\n");
	}

	MM_TYPE *mm = readMMfile(argv[1]);
	MY_CR_TYPE *cr = convertMMtoCRmatrix(mm);
	writeCR(cr,argv[2],atoi(argv[3]));


	return 0;
}




