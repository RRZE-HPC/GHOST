#define _XOPEN_SOURCE 500

#include "crs.h"
#include "ghost_mat.h"
#include "ghost_util.h"

#include <unistd.h>
#include <sys/types.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <byteswap.h>

#include <dlfcn.h>

#define CR(mat) ((CR_TYPE *)(mat->data))

const char name[] = "CRS plugin for ghost";
const char version[] = "0.1a";
const char formatID[] = "CRS";

static ghost_mnnz_t CRS_nnz(ghost_mat_t *mat);
static ghost_midx_t CRS_nrows(ghost_mat_t *mat);
static ghost_midx_t CRS_ncols(ghost_mat_t *mat);
static void CRS_fromBin(ghost_mat_t *mat, char *matrixPath);
static void CRS_fromMM(ghost_mat_t *mat, char *matrixPath);
static void CRS_printInfo(ghost_mat_t *mat);
static char * CRS_formatName(ghost_mat_t *mat);
static ghost_midx_t CRS_rowLen (ghost_mat_t *mat, ghost_midx_t i);
static ghost_mdat_t CRS_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j);
static size_t CRS_byteSize (ghost_mat_t *mat);
static void CRS_free(ghost_mat_t * mat);
static void CRS_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
static void CRS_fromCRS(ghost_mat_t *mat, void *crs);
static void CRS_readRpt(void *arg);
static void CRS_readColValOffset(void *args);
static void CRS_readHeader(void *vargs);
static void CRS_upload(ghost_mat_t *mat);
static int compareNZEPos( const void* a, const void* b ); 
#ifdef OPENCL
static void CRS_kernel_CL (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif

static int swapReq = 0;


ghost_mat_t *init(ghost_mtraits_t *traits)
{
	ghost_mat_t *mat = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	mat->traits = traits;

	DEBUG_LOG(1,"Initializing CRS functions");

	mat->fromBin = &CRS_fromBin;
	mat->fromMM = &CRS_fromMM;
	mat->fromCRS = &CRS_fromCRS;
	mat->printInfo = &CRS_printInfo;
	mat->formatName = &CRS_formatName;
	mat->rowLen   = &CRS_rowLen;
//	mat->entry    = &CRS_entry;
	mat->byteSize = &CRS_byteSize;
	mat->nnz      = &CRS_nnz;
	mat->nrows    = &CRS_nrows;
	mat->ncols    = &CRS_ncols;
	mat->destroy  = &CRS_free;
	mat->CLupload = &CRS_upload;
	mat->extraFun = (ghost_dummyfun_t *)allocateMemory(3*sizeof(ghost_dummyfun_t),"dummyfun CRS");
	mat->extraFun[GHOST_CRS_EXTRAFUN_READ_RPT] = &CRS_readRpt;
	mat->extraFun[GHOST_CRS_EXTRAFUN_READ_COL_VAL_OFFSET] = &CRS_readColValOffset;
	mat->extraFun[GHOST_CRS_EXTRAFUN_READ_HEADER] = &CRS_readHeader;
#ifdef OPENCL
	if (traits->flags & GHOST_SPM_HOST)
		mat->kernel   = &CRS_kernel_plain;
	else 
		mat->kernel   = &CRS_kernel_CL;
#else
	mat->kernel   = &CRS_kernel_plain;
#endif
	mat->data = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "CR(mat)" );
	mat->rowPerm = NULL;
	mat->invRowPerm = NULL;
	return mat;

}

static ghost_mnnz_t CRS_nnz(ghost_mat_t *mat)
{
	return CR(mat)->nEnts;
}
static ghost_midx_t CRS_nrows(ghost_mat_t *mat)
{
	return CR(mat)->nrows;
}
static ghost_midx_t CRS_ncols(ghost_mat_t *mat)
{
	return CR(mat)->ncols;
}

static void CRS_printInfo(ghost_mat_t *mat)
{
	UNUSED(mat);
	return;
}

static char * CRS_formatName(ghost_mat_t *mat)
{
	UNUSED(mat);
	return "CRS";
}

static ghost_midx_t CRS_rowLen (ghost_mat_t *mat, ghost_midx_t i)
{
	return CR(mat)->rpt[i+1] - CR(mat)->rpt[i];
}

static ghost_mdat_t CRS_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j)
{
	ghost_midx_t e;
	for (e=CR(mat)->rpt[i]; e<CR(mat)->rpt[i+1]; e++) {
		if (CR(mat)->col[e] == j)
			return CR(mat)->val[e];
	}
	return 0.;
}

static size_t CRS_byteSize (ghost_mat_t *mat)
{
	return (size_t)((CR(mat)->nrows+1)*sizeof(ghost_mnnz_t) + 
			CR(mat)->nEnts*(sizeof(ghost_midx_t)+sizeof(ghost_mdat_t)));
}


static void CRS_fromCRS(ghost_mat_t *mat, void *crs)
{
	DEBUG_LOG(1,"Creating CRS matrix");
	CR_TYPE *cr = (CR_TYPE*)crs;
	ghost_midx_t i,j;


	mat->data = (CR_TYPE *)allocateMemory(sizeof(CR_TYPE),"CR(mat)");
	CR(mat)->nrows = cr->nrows;
	CR(mat)->ncols = cr->ncols;
	CR(mat)->nEnts = cr->nEnts;

	CR(mat)->rpt = (ghost_midx_t *)allocateMemory((cr->nrows+1)*sizeof(ghost_midx_t),"rpt");
	CR(mat)->col = (ghost_midx_t *)allocateMemory(cr->nEnts*sizeof(ghost_midx_t),"col");
	CR(mat)->val = (ghost_mdat_t *)allocateMemory(cr->nEnts*sizeof(ghost_mdat_t),"val");

#pragma omp parallel for schedule(runtime)
	for( i = 0; i < CR(mat)->nrows+1; i++ ) {
		CR(mat)->rpt[i] = cr->rpt[i];
	}

#pragma omp parallel for schedule(runtime) private(j)
	for( i = 0; i < CR(mat)->nrows; i++ ) {
		for(j = CR(mat)->rpt[i]; j < CR(mat)->rpt[i+1] ; j++) {
			CR(mat)->col[j] = cr->col[j];
			CR(mat)->val[j] = cr->val[j];
		}
	}

	// TODO OpenCL upload

}



static void CRS_readHeader(void *vargs)
{
	CRS_readRpt_args_t *args = (CRS_readRpt_args_t *)vargs;
	ghost_mat_t * mat = args->mat;
	char *matrixPath = args->matrixPath;
	FILE* file;
	long filesize;
	int32_t endianess;
	int32_t fileVersion;
	int32_t base;
	int32_t symmetry;
	int32_t datatype;

	DEBUG_LOG(1,"Reading header from %s",matrixPath);

	if ((file = fopen(matrixPath, "rb"))==NULL){
		ABORT("Could not open binary CRS file %s",matrixPath);
	}

	fseek(file,0L,SEEK_END);
	filesize = ftell(file);
	fseek(file,0L,SEEK_SET);



	fread(&endianess, 4, 1, file);
	//	if (endianess != GHOST_BINCRS_LITTLE_ENDIAN)
	//		ABORT("Big endian currently not supported!");
	if (endianess == GHOST_BINCRS_LITTLE_ENDIAN && ghost_archIsBigEndian()) {
		DEBUG_LOG(1,"Need to convert from little to big endian.");
		swapReq = 1;
	} else if (endianess != GHOST_BINCRS_LITTLE_ENDIAN && !ghost_archIsBigEndian()) {
		DEBUG_LOG(1,"Need to convert from big to little endian.");
		swapReq = 1;
	} else {
		DEBUG_LOG(1,"OK, file and library have same endianess.");
	}

	fread(&fileVersion, 4, 1, file);
	if (swapReq) fileVersion = bswap_32(fileVersion);
	if (fileVersion != 1)
		ABORT("Can not read version %d of binary CRS format!",fileVersion);

	fread(&base, 4, 1, file);
	if (swapReq) base = bswap_32(base);
	if (base != 0)
		ABORT("Can not read matrix with %d-based indices!",base);

	fread(&symmetry, 4, 1, file);
	if (swapReq) symmetry = bswap_32(symmetry);
	if (!ghost_symmetryValid(symmetry))
		ABORT("Symmetry is invalid! (%d)",symmetry);
	if (symmetry != GHOST_BINCRS_SYMM_GENERAL)
		ABORT("Can not handle symmetry different to general at the moment!");
	mat->symmetry = symmetry;

	fread(&datatype, 4, 1, file);
	if (swapReq) datatype = bswap_32(datatype);
	if (!ghost_datatypeValid(datatype))
		ABORT("Datatype is invalid! (%d)",datatype);

	int64_t nr, nc, ne;

	fread(&nr, 8, 1, file);
	if (swapReq)  nr  = bswap_64(nr);
	CR(mat)->nrows = (ghost_midx_t)nr;

	fread(&nc, 8, 1, file);
	if (swapReq)  nc  = bswap_64(nc);
	CR(mat)->ncols = (ghost_midx_t)nc;

	fread(&ne, 8, 1, file);
	if (swapReq)  ne  = bswap_64(ne);
	CR(mat)->nEnts = (ghost_midx_t)ne;

	DEBUG_LOG(1,"Matrix has %d rows, %d columns and %d nonzeros",CR(mat)->nrows,CR(mat)->ncols,CR(mat)->nEnts);

	long rightFilesize = GHOST_BINCRS_SIZE_HEADER +
		(long)(CR(mat)->nrows+1) * GHOST_BINCRS_SIZE_RPT_EL +
		(long)CR(mat)->nEnts * GHOST_BINCRS_SIZE_COL_EL +
		(long)CR(mat)->nEnts * ghost_sizeofDataType(datatype);

	if (filesize != rightFilesize)
		ABORT("File has invalid size! (is: %ld, should be: %ld)",filesize, rightFilesize);

	DEBUG_LOG(1,"CRS matrix has %"PRmatIDX" rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros",CR(mat)->nrows,CR(mat)->ncols,CR(mat)->nEnts);

	fclose(file);
}

static void CRS_readRpt(void *vargs)
{
	CRS_readRpt_args_t *args = (CRS_readRpt_args_t *)vargs;
	ghost_mat_t * mat = args->mat;
	char *matrixPath = args->matrixPath;
	int file;
	ghost_midx_t i;

	DEBUG_LOG(1,"Reading row pointers from %s",matrixPath);

	if ((file = open(matrixPath, O_RDONLY)) == -1){
		ABORT("Could not open binary CRS file %s",matrixPath);
	}

	DEBUG_LOG(2,"Allocate memory for CR(mat)->rpt");
	CR(mat)->rpt = (ghost_midx_t *)    allocateMemory( (CR(mat)->nrows+1)*sizeof(ghost_midx_t), "rpt" );

	DEBUG_LOG(1,"NUMA-placement for CR(mat)->rpt");
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < CR(mat)->nrows+1; i++ ) {
		CR(mat)->rpt[i] = 0;
	}

	DEBUG_LOG(2,"Reading array with row-offsets");
#ifdef LONGIDX
	if (swapReq) {
		int64_t tmp;
		for( i = 0; i < CR(mat)->nrows+1; i++ ) {
			pread(file,&tmp, GHOST_BINCRS_SIZE_RPT_EL, GHOST_BINCRS_SIZE_HEADER+i*8);
			tmp = bswap_64(tmp);
			CR(mat)->rpt[i] = tmp;
		}
	} else {
		pread(file,&CR(mat)->rpt[0], GHOST_BINCRS_SIZE_RPT_EL*(CR(mat)->nrows+1), GHOST_BINCRS_SIZE_HEADER);
	}
#else // casting
	DEBUG_LOG(1,"Casting from 64 bit to 32 bit row pointers");
	int64_t *tmp = (int64_t *)malloc((CR(mat)->nrows+1)*8);
	pread(file,tmp, GHOST_BINCRS_SIZE_COL_EL*(CR(mat)->nrows+1), GHOST_BINCRS_SIZE_HEADER );

	if (swapReq) {
		for( i = 0; i < CR(mat)->nrows+1; i++ ) {
			CR(mat)->rpt[i] = (ghost_midx_t)(bswap_64(tmp[i]));
		}
	} else {
		for( i = 0; i < CR(mat)->nrows+1; i++ ) {
			CR(mat)->rpt[i] = (ghost_midx_t)(tmp[i]);
		}
	}
	free(tmp);
#endif
}

static void CRS_readColValOffset(void *vargs)
{
	CRS_readColValOffset_args_t *args = (CRS_readColValOffset_args_t *)vargs;
	ghost_mat_t *mat = args->mat;
	char *matrixPath = args->matrixPath;
	ghost_mnnz_t offsetEnts = args->offsetEnts;
	ghost_midx_t offsetRows = args->offsetRows;
	ghost_mnnz_t nEnts = args->nEnts;
	ghost_midx_t nRows = args->nRows;
	int IOtype = args->IOtype;


	UNUSED(offsetRows);	
	UNUSED(IOtype);

	ghost_midx_t i, j;
	int datatype;
	int file;

	off_t offs;

	file = open(matrixPath,O_RDONLY);

	DEBUG_LOG(1,"Reading %"PRmatNNZ" cols and vals from binary file %s with offset %"PRmatNNZ,nEnts, matrixPath,offsetEnts);

	if ((file = open(matrixPath, O_RDONLY)) == -1){
		ABORT("Could not open binary CRS file %s",matrixPath);
	}
	pread(file, &datatype, sizeof(int), 16);
	if (swapReq) datatype = bswap_32(datatype);

	DEBUG_LOG(1,"CRS matrix has %"PRmatIDX" rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros",CR(mat)->nrows,CR(mat)->ncols,CR(mat)->nEnts);

	DEBUG_LOG(2,"Allocate memory for CR(mat)->col and CR(mat)->val");
	CR(mat)->col       = (ghost_midx_t *) allocateMemory( nEnts * sizeof(ghost_midx_t),  "col" );
	CR(mat)->val       = (ghost_mdat_t *) allocateMemory( nEnts * sizeof(ghost_mdat_t),  "val" );

	DEBUG_LOG(2,"NUMA-placement for CR(mat)->val and CR(mat)->col");
#pragma omp parallel for schedule(runtime) private(j)
	for(i = 0 ; i < nRows; ++i) {
		for(j = CR(mat)->rpt[i]; j < CR(mat)->rpt[i+1] ; j++) {
			CR(mat)->val[j] = 0.0;
			CR(mat)->col[j] = 0;
		}
	}


	DEBUG_LOG(1,"Reading array with column indices");
	offs = GHOST_BINCRS_SIZE_HEADER+
		GHOST_BINCRS_SIZE_RPT_EL*(CR(mat)->nrows+1)+
		GHOST_BINCRS_SIZE_COL_EL*offsetEnts;
#ifdef LONGIDX
	if (swapReq) {
		int64_t *tmp = (int64_t *)malloc(nEnts*8);
		pread(file,tmp, GHOST_BINCRS_SIZE_COL_EL*nEnts, offs );
		for( i = 0; i < nEnts; i++ ) {
			CR(mat)->col[i] = bswap_64(tmp[i]);
		}
	} else {
		pread(file,&CR(mat)->col[0], GHOST_BINCRS_SIZE_COL_EL*nEnts, offs );
	}
#else // casting
	DEBUG_LOG(1,"Casting from 64 bit to 32 bit column indices");
	int64_t *tmp = (int64_t *)malloc(nEnts*8);
	pread(file,tmp, GHOST_BINCRS_SIZE_COL_EL*nEnts, offs );
	for(i = 0 ; i < nRows; ++i) {
		for(j = CR(mat)->rpt[i]; j < CR(mat)->rpt[i+1] ; j++) {
			if (swapReq) CR(mat)->col[j] = (ghost_midx_t)(bswap_64(tmp[j]));
			else CR(mat)->col[j] = (ghost_midx_t)tmp[j];
		}
	}
	free(tmp);
#endif
		// minimal size of value
		size_t valSize = sizeof(float);
		if (datatype & GHOST_BINCRS_DT_DOUBLE)
			valSize *= 2;

		if (datatype & GHOST_BINCRS_DT_COMPLEX)
			valSize *= 2;


	DEBUG_LOG(1,"Reading array with values");
	offs = GHOST_BINCRS_SIZE_HEADER+
		GHOST_BINCRS_SIZE_RPT_EL*(CR(mat)->nrows+1)+
		GHOST_BINCRS_SIZE_COL_EL*CR(mat)->nEnts+
		ghost_sizeofDataType(datatype)*offsetEnts;

	if (datatype == GHOST_MY_MDATATYPE) {
		if (swapReq) {
		uint8_t *tmpval = (uint8_t *)allocateMemory(nEnts*valSize,"tmpval");
		pread(file,tmpval, nEnts*valSize, offs);
		if (GHOST_MY_MDATATYPE & GHOST_BINCRS_DT_COMPLEX) {
			if (GHOST_MY_MDATATYPE & GHOST_BINCRS_DT_FLOAT) {
				for (i = 0; i<nEnts; i++) {
					uint32_t *a = (uint32_t *)tmpval;
					uint32_t rswapped = bswap_32(a[2*i]);
					uint32_t iswapped = bswap_32(a[2*i+1]);
					memcpy(&(CR(mat)->val[i]),&rswapped,4);
					memcpy(&(CR(mat)->val[i])+4,&iswapped,4);
				}
			} else {
				for (i = 0; i<nEnts; i++) {
					uint64_t *a = (uint64_t *)tmpval;
					uint64_t rswapped = bswap_64(a[2*i]);
					uint64_t iswapped = bswap_64(a[2*i+1]);
					memcpy(&(CR(mat)->val[i]),&rswapped,8);
					memcpy(&(CR(mat)->val[i])+8,&iswapped,8);
				}
			}
		} else {
			if (GHOST_MY_MDATATYPE & GHOST_BINCRS_DT_FLOAT) {
				for (i = 0; i<nEnts; i++) {
					uint32_t *a = (uint32_t *)tmpval;
					uint32_t swapped = bswap_32(a[i]);
					memcpy(&(CR(mat)->val[i]),&swapped,4);
				}
			} else {
				for (i = 0; i<nEnts; i++) {
					uint64_t *a = (uint64_t *)tmpval;
					uint64_t swapped = bswap_64(a[i]);
					memcpy(&(CR(mat)->val[i]),&swapped,8);
				}
			}

		}
		} else {
			pread(file,&CR(mat)->val[0], ghost_sizeofDataType(datatype)*nEnts, offs );
		}
	} else {

		WARNING_LOG("This %s build is configured for %s data but"
				" the file contains %s data. Casting...",GHOST_NAME,
				ghost_datatypeName(GHOST_MY_MDATATYPE),ghost_datatypeName(datatype));


		uint8_t *tmpval = (uint8_t *)allocateMemory(nEnts*valSize,"tmpval");
		pread(file,tmpval, nEnts*valSize, offs);

		if (swapReq) {
			ABORT("Not yet supported!");
			if (GHOST_MY_MDATATYPE & GHOST_BINCRS_DT_COMPLEX) {
				if (GHOST_MY_MDATATYPE & GHOST_BINCRS_DT_FLOAT) {
					for (i = 0; i<nEnts; i++) {
						CR(mat)->val[i] = (ghost_mdat_t) ((bswap_32(tmpval[i*valSize]))+
							I*(bswap_32(tmpval[i*valSize+valSize/2])));
					}
				} else {
					for (i = 0; i<nEnts; i++) {
						CR(mat)->val[i] = (ghost_mdat_t) ((bswap_64(tmpval[i*valSize]))+
							I*(bswap_64(tmpval[i*valSize+valSize/2])));
					}
				}
			} else {
				if (GHOST_MY_MDATATYPE & GHOST_BINCRS_DT_FLOAT) {
					for (i = 0; i<nEnts; i++) {
						CR(mat)->val[i] = (ghost_mdat_el_t) (bswap_32(tmpval[i*valSize]));
					}
				} else {
					for (i = 0; i<nEnts; i++) {
						CR(mat)->val[i] = (ghost_mdat_el_t) (bswap_64(tmpval[i*valSize]));
					}
				}

			}

		} else {
			for (i = 0; i<nEnts; i++) CR(mat)->val[i] = (ghost_mdat_t) tmpval[i*valSize];
		}

		free(tmpval);
	}
	close(file);




}

static void CRS_upload(ghost_mat_t *mat)
{
	DEBUG_LOG(1,"Uploading CRS matrix to device");
#ifdef OPENCL
	if (!(mat->traits->flags & GHOST_SPM_HOST)) {
		DEBUG_LOG(1,"Creating matrix on OpenCL device");
		CR(mat)->clmat = (CL_CR_TYPE *)allocateMemory(sizeof(CL_CR_TYPE),"CL_CRS");
		CR(mat)->clmat->rpt = CL_allocDeviceMemory((CR(mat)->nrows+1)*sizeof(ghost_cl_mnnz_t));
		CR(mat)->clmat->col = CL_allocDeviceMemory((CR(mat)->nEnts)*sizeof(ghost_cl_midx_t));
		CR(mat)->clmat->val = CL_allocDeviceMemory((CR(mat)->nEnts)*sizeof(ghost_cl_mdat_t));

		CR(mat)->clmat->nrows = CR(mat)->nrows;
		CL_copyHostToDevice(CR(mat)->clmat->rpt, CR(mat)->rpt, (CR(mat)->nrows+1)*sizeof(ghost_cl_mnnz_t));
		CL_copyHostToDevice(CR(mat)->clmat->col, CR(mat)->col, CR(mat)->nEnts*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(CR(mat)->clmat->val, CR(mat)->val, CR(mat)->nEnts*sizeof(ghost_cl_mdat_t));

		cl_int err;
		cl_uint numKernels;
		cl_program program = CL_registerProgram("crs_clkernel.cl","");
		CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
		DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
		mat->clkernel = clCreateKernel(program,"CRS_kernel",&err);
		CL_checkerror(err);

		CL_safecall(clSetKernelArg(mat->clkernel,3,sizeof(int), &(CR(mat)->clmat->nrows)));
		CL_safecall(clSetKernelArg(mat->clkernel,4,sizeof(cl_mem), &(CR(mat)->clmat->rpt)));
		CL_safecall(clSetKernelArg(mat->clkernel,5,sizeof(cl_mem), &(CR(mat)->clmat->col)));
		CL_safecall(clSetKernelArg(mat->clkernel,6,sizeof(cl_mem), &(CR(mat)->clmat->val)));
	}
#else
	if (mat->traits->flags & GHOST_SPM_DEVICE) {
		ABORT("Device matrix cannot be created without OpenCL");
	}
#endif
}

int compareNZEPos( const void* a, const void* b ) 
{

	/* comparison function for sorting of matrix entries;
	 * sort lesser row id first, then lesser column id first;
	 * if MAIN_DIAGONAL_FIRST is defined sort diagonal 
	 * before lesser column id */

	int aRow = ((NZE_TYPE*)a)->row,
	    bRow = ((NZE_TYPE*)b)->row,
	    aCol = ((NZE_TYPE*)a)->col,
	    bCol = ((NZE_TYPE*)b)->col;

	if( aRow == bRow ) {
#ifdef MAIN_DIAGONAL_FIRST
		if( aRow == aCol ) aCol = -1;
		if( bRow == bCol ) bCol = -1;
#endif /* MAIN_DIAGONAL_FIRST */
		return aCol - bCol;
	}
	else return aRow - bRow;
}

static void CRS_fromBin(ghost_mat_t *mat, char *matrixPath)
{

	/*	int detectDiags = 0;
		ghost_midx_t i, j;
		int datatype;
		int file;
		file = open(matrixPath,O_RDONLY);

		mat->data = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "CR(mat)" );
		mat->rowPerm = NULL;
		mat->invRowPerm = NULL;

		DEBUG_LOG(1,"Reading binary CRS matrix %s",matrixPath);

		if ((file = open(matrixPath, O_RDONLY)) == -1){
		ABORT("Could not open binary CRS file %s",matrixPath);
		}

		pread(file, &datatype, sizeof(int), 0);
		pread(file, &CR(mat)->nrows, sizeof(int), 1*sizeof(int));
		pread(file, &CR(mat)->ncols, sizeof(int), 2*sizeof(int));
		pread(file, &CR(mat)->nEnts, sizeof(int), 3*sizeof(int));

		DEBUG_LOG(1,"CRS matrix has %"PRmatIDX" rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros",CR(mat)->nrows,CR(mat)->ncols,CR(mat)->nEnts);

		if (datatype != GHOST_MY_MDATATYPE) {
		DEBUG_LOG(0,"Warning in %s:%d! The library has been built for %s data but"
		" the file contains %s data. Casting...\n",__FILE__,__LINE__,
		ghost_datatypeName(GHOST_MY_MDATATYPE),ghost_datatypeName(datatype));
		}

		DEBUG_LOG(2,"Allocate memory for CR(mat)->rpt");
		CR(mat)->rpt = (ghost_midx_t *)    allocateMemory( (CR(mat)->nrows+1)*sizeof(ghost_midx_t), "rpt" );

		DEBUG_LOG(1,"NUMA-placement for CR(mat)->rpt");
#pragma omp parallel for schedule(runtime)
for( i = 0; i < CR(mat)->nrows+1; i++ ) {
CR(mat)->rpt[i] = 0;
}

DEBUG_LOG(2,"Reading array with row-offsets");
pread(file,&CR(mat)->rpt[0], sizeof(int)*(CR(mat)->nrows+1), 4*sizeof(int));


CR(mat)->constDiags = NULL;

if (detectDiags) {
ghost_midx_t bandwidth = 2;//CR(mat)->ncols/2;
ghost_midx_t nDiags = 2*bandwidth + 1;

ghost_mdat_t *diagVals = (ghost_mdat_t *)allocateMemory(nDiags*sizeof(ghost_mdat_t),"diagVals");

char *diagStatus = (char *)allocateMemory(nDiags*sizeof(char),"diagStatus");
for (i=0; i<nDiags; i++) diagStatus[i] = DIAG_NEW;

int *diagEnts = (int *)allocateMemory(nDiags*sizeof(int),"diagEnts");
for (i=0; i<nDiags; i++) diagEnts[i] = 0;

DEBUG_LOG(1,"Detecting constant subdiagonals within a band of width %"PRmatIDX,bandwidth);
ghost_midx_t *tmpcol = (ghost_midx_t *)allocateMemory(CR(mat)->nEnts*sizeof(ghost_midx_t),"tmpcol");
ghost_mdat_t *tmpval = (ghost_mdat_t *)allocateMemory(CR(mat)->nEnts*sizeof(ghost_mdat_t),"tmpval");

int pfile;
pfile = open(matrixPath,O_RDONLY);
int offs = 4*sizeof(int)+(CR(mat)->nrows+1)*sizeof(int);
int idx = 0;
for (i=0; i<CR(mat)->nrows; ++i) {
for(j = CR(mat)->rpt[i] ; j < CR(mat)->rpt[i+1] ; j++) {
pread(pfile,&tmpcol[idx],sizeof(int),offs+idx*sizeof(int));
pread(pfile,&tmpval[idx],sizeof(ghost_mdat_t),offs+CR(mat)->nEnts*sizeof(int)+idx*sizeof(ghost_mdat_t));
if ((ghost_midx_t)llabs(tmpcol[idx]-i) <= bandwidth) { // in band

int didx = tmpcol[idx]-i+bandwidth; // index of diagonal
	if (diagStatus[didx] == DIAG_NEW) { // first time
		diagVals[didx] =  tmpval[idx];
		diagStatus[didx] = (char)1;
		diagEnts[didx] = 1;
		DEBUG_LOG(2,"Diag %d initialized with %f+%fi",didx,REAL(diagVals[didx]),IMAG(diagVals[didx]));
	} else if (diagStatus[didx] == DIAG_OK) { // diag initialized
		if (EQUALS(diagVals[didx],tmpval[idx])) {
			diagEnts[didx]++;
		} else {
			DEBUG_LOG(2,"Diag %d discontinued in row %"PRmatIDX": %f+%fi (was %f%fi)",didx,i,REAL(tmpval[idx]),IMAG(tmpval[idx]),REAL(diagVals[didx]),IMAG(diagVals[didx]));
			diagStatus[didx] = DIAG_INVALID;
		}
	}
}
idx++;
}
}

CR(mat)->nConstDiags = 0;

for (i=0; i<bandwidth+1; i++) { // lower subdiagonals AND diagonal
	if (diagStatus[i] == DIAG_OK && diagEnts[i] == CR(mat)->ncols-bandwidth+i) {
		DEBUG_LOG(1,"The %"PRmatIDX"-th subdiagonal is constant with %f+%fi",bandwidth-i,REAL(diagVals[i]),IMAG(diagVals[i]));
		CR(mat)->nConstDiags++;
		CR(mat)->constDiags = realloc(CR(mat)->constDiags,sizeof(CONST_DIAG)*CR(mat)->nConstDiags);
		CR(mat)->constDiags[CR(mat)->nConstDiags-1].idx = bandwidth-i;
		CR(mat)->constDiags[CR(mat)->nConstDiags-1].val = diagVals[i];
		CR(mat)->constDiags[CR(mat)->nConstDiags-1].len = diagEnts[i];
		CR(mat)->constDiags[CR(mat)->nConstDiags-1].minRow = i-bandwidth;
		CR(mat)->constDiags[CR(mat)->nConstDiags-1].maxRow = CR(mat)->nrows-1;
		DEBUG_LOG(1,"range: %"PRmatIDX"..%"PRmatIDX,i-bandwidth,CR(mat)->nrows-1);
	}
}
for (i=bandwidth+1; i<nDiags ; i++) { // upper subdiagonals
	if (diagStatus[i] == DIAG_OK && diagEnts[i] == CR(mat)->ncols+bandwidth-i) {
		DEBUG_LOG(1,"The %"PRmatIDX"-th subdiagonal is constant with %f+%fi",-bandwidth+i,REAL(diagVals[i]),IMAG(diagVals[i]));
		CR(mat)->nConstDiags++;
		CR(mat)->constDiags = realloc(CR(mat)->constDiags,sizeof(CONST_DIAG)*CR(mat)->nConstDiags);
		CR(mat)->constDiags[CR(mat)->nConstDiags-1].idx = -bandwidth+i;
		CR(mat)->constDiags[CR(mat)->nConstDiags-1].val = diagVals[i];
		CR(mat)->constDiags[CR(mat)->nConstDiags-1].len = diagEnts[i];
		CR(mat)->constDiags[CR(mat)->nConstDiags-1].minRow = 0;
		CR(mat)->constDiags[CR(mat)->nConstDiags-1].maxRow = CR(mat)->nrows-1-i+bandwidth;
		DEBUG_LOG(1,"range: 0..%"PRmatIDX,CR(mat)->nrows-1-i+bandwidth);
	}
}

if (CR(mat)->nConstDiags == 0) 
{ // no constant diagonals found
	CR(mat)->val = tmpval;
	CR(mat)->col = tmpcol;
} 
else {
	ghost_midx_t d = 0;

	DEBUG_LOG(1,"Adjusting the number of matrix entries, old: %"PRmatNNZ,CR(mat)->nEnts);
	for (d=0; d<CR(mat)->nConstDiags; d++) {
		CR(mat)->nEnts -= CR(mat)->constDiags[d].len;
	}
	DEBUG_LOG(1,"Adjusting the number of matrix entries, new: %"PRmatNNZ,CR(mat)->nEnts);

	DEBUG_LOG(2,"Allocate memory for CR(mat)->col and CR(mat)->val");
	CR(mat)->col       = (ghost_midx_t*)    allocateMemory( CR(mat)->nEnts * sizeof(ghost_midx_t),  "col" );
	CR(mat)->val       = (ghost_mdat_t*) allocateMemory( CR(mat)->nEnts * sizeof(ghost_mdat_t),  "val" );

	//TODO NUMA
	ghost_midx_t *newRowOffset = (ghost_midx_t *)allocateMemory((CR(mat)->nrows+1)*sizeof(ghost_midx_t),"newRowOffset");

	idx = 0;
	ghost_midx_t oidx = 0; // original idx in tmp arrays
	for (i=0; i<CR(mat)->nrows; ++i) {
		newRowOffset[i] = idx;
		for(j = CR(mat)->rpt[i] ; j < CR(mat)->rpt[i+1]; j++) {
			if ((ghost_midx_t)llabs(tmpcol[oidx]-i) <= bandwidth) { // in band
				int diagFound = 0;
				for (d=0; d<CR(mat)->nConstDiags; d++) {
					if (tmpcol[oidx]-i == CR(mat)->constDiags[d].idx) { // do not store constant diagonal
						oidx++;
						diagFound = 1;
						break;
					} 
				}
				if (!diagFound) {
					CR(mat)->col[idx] = tmpcol[oidx];
					CR(mat)->val[idx] = tmpval[oidx];
					idx++;
					oidx++;
				}

			} else {
				CR(mat)->col[idx] = tmpcol[oidx];
				CR(mat)->val[idx] = tmpval[oidx];
				idx++;
				oidx++;

			}
		}
	}
	free(CR(mat)->rpt);
	free(tmpval);
	free(tmpcol);

	newRowOffset[CR(mat)->nrows] = CR(mat)->nEnts;
	CR(mat)->rpt = newRowOffset;
}
close(pfile);
} 
else {

	DEBUG_LOG(2,"Allocate memory for CR(mat)->col and CR(mat)->val");
	CR(mat)->col       = (ghost_midx_t *)    allocateMemory( CR(mat)->nEnts * sizeof(ghost_midx_t),  "col" );
	CR(mat)->val       = (ghost_mdat_t *) allocateMemory( CR(mat)->nEnts * sizeof(ghost_mdat_t),  "val" );

	DEBUG_LOG(1,"NUMA-placement for CR(mat)->val and CR(mat)->col");
#pragma omp parallel for schedule(runtime)
	for(i = 0 ; i < CR(mat)->nrows; ++i) {
		for(j = CR(mat)->rpt[i] ; j < CR(mat)->rpt[i+1] ; j++) {
			CR(mat)->val[j] = 0.0;
			CR(mat)->col[j] = 0;
		}
	}


	DEBUG_LOG(2,"Reading array with column indices");

	pread(file,&CR(mat)->col[0], sizeof(int)*CR(mat)->nEnts, sizeof(int)*(CR(mat)->nrows+1+4));
	DEBUG_LOG(2,"Reading array with values");
	if (datatype == GHOST_MY_MDATATYPE)
	{
		pread(file,&CR(mat)->val[0], sizeof(ghost_mdat_t)*CR(mat)->nEnts, sizeof(int)*(CR(mat)->nrows+1+4+CR(mat)->nEnts));
	} 
	else 
	{*/
		/*	switch (datatype) {
			case GHOST_DATATYPE_S:
			{
			float *tmp = (float *)allocateMemory(
			CR(mat)->nEnts*sizeof(float), "tmp");
			pread(tmp, sizeof(float)*CR(mat)->nEnts, RESTFILE);
			for (i = 0; i<CR(mat)->nEnts; i++) CR(mat)->val[i] = (ghost_mdat_t) tmp[i];
			free(tmp);
			break;
			}
			case GHOST_DATATYPE_D:
			{
			double *tmp = (double *)allocateMemory(
			CR(mat)->nEnts*sizeof(double), "tmp");
			fread(tmp, sizeof(double), CR(mat)->nEnts, RESTFILE);
			for (i = 0; i<CR(mat)->nEnts; i++) CR(mat)->val[i] = (ghost_mdat_t) tmp[i];
			free(tmp);
			break;
			}
			case GHOST_DATATYPE_C:
			{
			_Complex float *tmp = (_Complex float *)allocateMemory(
			CR(mat)->nEnts*sizeof(_Complex float), "tmp");
			fread(tmp, sizeof(_Complex float), CR(mat)->nEnts, RESTFILE);
			for (i = 0; i<CR(mat)->nEnts; i++) CR(mat)->val[i] = (ghost_mdat_t) tmp[i];
			free(tmp);
			break;
			}
			case GHOST_DATATYPE_Z:
			{
			_Complex double *tmp = (_Complex double *)allocateMemory(
			CR(mat)->nEnts*sizeof(_Complex double), "tmp");
			fread(tmp, sizeof(_Complex double), CR(mat)->nEnts, RESTFILE);
			for (i = 0; i<CR(mat)->nEnts; i++) CR(mat)->val[i] = (ghost_mdat_t) tmp[i];
			free(tmp);
			break;
			}
			}*/
		/*	}
			}
			close(file);*/

	CRS_readRpt_args_t args = {.mat=mat,.matrixPath=matrixPath};
CRS_readHeader(&args);  // read header
CRS_readRpt(&args);

CRS_readColValOffset_args_t cvargs = {
	.mat=mat,
	.matrixPath=matrixPath,
	.nEnts = CR(mat)->nEnts,
	.offsetEnts = 0,
	.offsetRows = 0,
	.nRows = CR(mat)->nrows,
	.IOtype = GHOST_IO_STD};
CRS_readColValOffset(&cvargs);

/*

#ifdef OPENCL
if (!(mat->traits->flags & GHOST_SPM_HOST)) {
DEBUG_LOG(1,"Creating matrix on OpenCL device");
CR(mat)->clmat = (CL_CR_TYPE *)allocateMemory(sizeof(CL_CR_TYPE),"CL_CRS");
CR(mat)->clmat->rpt = CL_allocDeviceMemory((CR(mat)->nrows+1)*sizeof(ghost_cl_mnnz_t));
CR(mat)->clmat->col = CL_allocDeviceMemory((CR(mat)->nEnts)*sizeof(ghost_cl_midx_t));
CR(mat)->clmat->val = CL_allocDeviceMemory((CR(mat)->nEnts)*sizeof(ghost_cl_mdat_t));

CR(mat)->clmat->nrows = CR(mat)->nrows;
CL_copyHostToDevice(CR(mat)->clmat->rpt, CR(mat)->rpt, (CR(mat)->nrows+1)*sizeof(ghost_cl_mnnz_t));
CL_copyHostToDevice(CR(mat)->clmat->col, CR(mat)->col, CR(mat)->nEnts*sizeof(ghost_cl_midx_t));
CL_copyHostToDevice(CR(mat)->clmat->val, CR(mat)->val, CR(mat)->nEnts*sizeof(ghost_cl_mdat_t));

cl_int err;
cl_uint numKernels;
cl_program program = CL_registerProgram("crs_clkernel.cl","");
CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
mat->clkernel = clCreateKernel(program,"CRS_kernel",&err);
CL_checkerror(err);

CL_safecall(clSetKernelArg(mat->clkernel,3,sizeof(int), &(CR(mat)->clmat->nrows)));
CL_safecall(clSetKernelArg(mat->clkernel,4,sizeof(cl_mem), &(CR(mat)->clmat->rpt)));
CL_safecall(clSetKernelArg(mat->clkernel,5,sizeof(cl_mem), &(CR(mat)->clmat->col)));
CL_safecall(clSetKernelArg(mat->clkernel,6,sizeof(cl_mem), &(CR(mat)->clmat->val)));
}
#else
if (mat->traits->flags & GHOST_SPM_DEVICE) {
ABORT("Device matrix cannot be created without OpenCL");
}
#endif

DEBUG_LOG(1,"Matrix read in successfully");*/
}

static void CRS_fromMM(ghost_mat_t *mat, char *matrixPath)
{
	/*ghost_mm_t * mm = readMMFile(matrixPath);

	ghost_midx_t* nEntsInRow;
	ghost_midx_t i, e, pos;

	size_t size_rpt, size_col, size_val, size_nEntsInRow;


	DEBUG_LOG(1,"Converting MM to CRS matrix");


	size_rpt  = (size_t)( (mm->nrows+1) * sizeof( ghost_midx_t ) );
	size_col        = (size_t)( mm->nEnts     * sizeof( ghost_midx_t ) );
	size_val        = (size_t)( mm->nEnts     * sizeof( ghost_mdat_t) );
	size_nEntsInRow = (size_t)(  mm->nrows    * sizeof( ghost_midx_t) );

	mat->data = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "CR(mat)" );

	CR(mat)->rpt = (ghost_midx_t*)     allocateMemory( size_rpt,    "rpt" );
	CR(mat)->col = (ghost_midx_t*)     allocateMemory( size_col,          "col" );
	CR(mat)->val = (ghost_mdat_t*)  allocateMemory( size_val,          "val" );
	nEntsInRow = (ghost_midx_t*)     allocateMemory( size_nEntsInRow,   "nEntsInRow" );


	CR(mat)->nrows = mm->nrows;
	CR(mat)->ncols = mm->ncols;
	CR(mat)->nEnts = mm->nEnts;
	for( i = 0; i < mm->nrows; i++ ) nEntsInRow[i] = 0;

	qsort( mm->nze, (size_t)(mm->nEnts), sizeof( NZE_TYPE ), compareNZEPos );

	for( e = 0; e < mm->nEnts; e++ ) nEntsInRow[mm->nze[e].row]++;

	pos = 0;
	CR(mat)->rpt[0] = pos;
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < mm->nrows; i++ ) {
		CR(mat)->rpt[i] = 0;
	}

	for( i = 0; i < mm->nrows; i++ ) {
		CR(mat)->rpt[i] = pos;
		pos += nEntsInRow[i];
	}
	CR(mat)->rpt[mm->nrows] = pos;

	for( i = 0; i < mm->nrows; i++ ) nEntsInRow[i] = 0;

#pragma omp parallel for schedule(runtime)
	for(i=0; i<CR(mat)->nrows; ++i) {
		ghost_midx_t start = CR(mat)->rpt[i];
		ghost_midx_t end = CR(mat)->rpt[i+1];
		ghost_midx_t j;
		for(j=start; j<end; j++) {
			CR(mat)->val[j] = 0.0;
			CR(mat)->col[j] = 0;
		}
	}

	for( e = 0; e < mm->nEnts; e++ ) {
		const int row = mm->nze[e].row,
		      col = mm->nze[e].col;
		const ghost_mdat_t val = mm->nze[e].val;
		pos = CR(mat)->rpt[row] + nEntsInRow[row];
		CR(mat)->col[pos] = col;

		CR(mat)->val[pos] = val;

		nEntsInRow[row]++;
	}
	free( nEntsInRow );

	DEBUG_LOG(1,"CR matrix created from MM successfully" );*/

}

static void CRS_free(ghost_mat_t * mat)
{
	DEBUG_LOG(1,"Freeing CRS matrix");
#ifdef OPENCL
	if (mat->traits->flags & GHOST_SPM_DEVICE) {
		CL_freeDeviceMemory(CR(mat)->clmat->rpt);
		CL_freeDeviceMemory(CR(mat)->clmat->col);
		CL_freeDeviceMemory(CR(mat)->clmat->val);
	}
#endif
	free(CR(mat)->rpt);
	free(CR(mat)->col);
	free(CR(mat)->val);

	free(mat->data);
	free(mat->rowPerm);
	free(mat->invRowPerm);
	free(mat->extraFun);


	free(mat);
}

static void CRS_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	/*	if (mat->symmetry == GHOST_BINCRS_SYMM_SYMMETRIC) {
		ghost_midx_t i, j;
		ghost_vdat_t hlp1;
		ghost_midx_t col;
		ghost_mdat_t val;

#pragma omp	parallel for schedule(runtime) private (hlp1, j, col, val)
for (i=0; i<CR(mat)->nrows; i++){
hlp1 = 0.0;

j = CR(mat)->rpt[i];

if (CR(mat)->col[j] == i) {
col = CR(mat)->col[j];
val = CR(mat)->val[j];

hlp1 += val * rhs->val[col];

j++;
} else {
printf("row %d has diagonal 0\n",i);
}


for (; j<CR(mat)->rpt[i+1]; j++){
col = CR(mat)->col[j];
val = CR(mat)->val[j];

hlp1 += val * rhs->val[col];

if (i!=col) {	
#pragma omp atomic
lhs->val[col] += val * rhs->val[i];  // FIXME non-axpy case maybe doesnt work
}

}
if (options & GHOST_SPMVM_AXPY) {
lhs->val[i] += hlp1;
} else {
lhs->val[i] = hlp1;
}
}

} else {*/


   double *rhsv = (double *)rhs->val;	
   double *lhsv = (double *)lhs->val;	
	ghost_midx_t i, j;
	ghost_vdat_t hlp1;
	CR_TYPE *cr = CR(mat);
	
#pragma omp parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<cr->nrows; i++){
		hlp1 = 0.0;
		for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
			hlp1 = hlp1 + (ghost_vdat_t)cr->val[j] * rhsv[cr->col[j]];
		}
		if (options & GHOST_SPMVM_AXPY) 
			lhsv[i] += hlp1;
		else
			lhsv[i] = hlp1;
	}

//}
}

#ifdef OPENCL
static void CRS_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	CL_safecall(clSetKernelArg(mat->clkernel,0,sizeof(cl_mem), &(lhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(mat->clkernel,1,sizeof(cl_mem), &(rhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(mat->clkernel,2,sizeof(int), &options));

	size_t gSize = (size_t)CR(mat)->clmat->nrows;

	CL_enqueueKernel(mat->clkernel,1,&gSize,NULL);
}
#endif
