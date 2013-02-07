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
static void CRS_fromBin(ghost_mat_t *mat, char *matrixPath, ghost_context_t *ctx, int options);
static void CRS_printInfo(ghost_mat_t *mat);
static char * CRS_formatName(ghost_mat_t *mat);
static ghost_midx_t CRS_rowLen (ghost_mat_t *mat, ghost_midx_t i);
//static ghost_mdat_t CRS_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j);
static size_t CRS_byteSize (ghost_mat_t *mat);
static void CRS_free(ghost_mat_t * mat);
static void CRS_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
static void CRS_fromCRS(ghost_mat_t *mat, void *crs);
static void CRS_readRpt(ghost_mat_t *mat, char *matrixPath);
static void CRS_readColValOffset(ghost_mat_t *mat, char *matrixPath, ghost_mnnz_t offsetEnts, ghost_midx_t offsetRows, ghost_midx_t nRows, ghost_mnnz_t nEnts, int IOtype);
static void CRS_readHeader(ghost_mat_t *mat, char *matrixPath);
#ifdef MPI
static void CRS_createDistribution(ghost_mat_t *mat, int options, ghost_comm_t *lcrp);
static void CRS_createCommunication(ghost_mat_t *mat, CR_TYPE **localCR, CR_TYPE **remoteCR, int options, ghost_context_t *context);
#endif
static void CRS_upload(ghost_mat_t *mat);
//static int compareNZEPos( const void* a, const void* b ); 
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
//	mat->fromMM = &CRS_fromMM;
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

/*static ghost_mdat_t CRS_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j)
{
	ghost_midx_t e;
	for (e=CR(mat)->rpt[i]; e<CR(mat)->rpt[i+1]; e++) {
		if (CR(mat)->col[e] == j)
			return CR(mat)->val[e];
	}
	return 0.;
}*/

static size_t CRS_byteSize (ghost_mat_t *mat)
{
	return (size_t)((CR(mat)->nrows+1)*sizeof(ghost_mnnz_t) + 
			CR(mat)->nEnts*(sizeof(ghost_midx_t)+sizeof(ghost_mdat_t)));
}


static void CRS_fromCRS(ghost_mat_t *mat, void *crs)
{
	DEBUG_LOG(1,"Creating CRS matrix from CRS matrix");
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

#ifdef MPI
static void CRS_createDistributedContext(ghost_context_t * context, char * matrixPath, int options, ghost_mtraits_t *traits)
{
	DEBUG_LOG(1,"Creating distributed context with parallel MPI-IO");

	ghost_midx_t i;

	/* Processor rank (MPI-process) */
	int me = ghost_getRank(); 

	ghost_comm_t *comm;

	ghost_mat_t *CRSfullMatrix;

	unsigned int nprocs = ghost_getNumberOfProcesses();

	/****************************************************************************
	 *******            ........ Executable statements ........           *******
	 ***************************************************************************/

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

	DEBUG_LOG(1,"Entering context_communication_parallel");

	comm = (ghost_comm_t*) allocateMemory( sizeof(ghost_comm_t), "comm");
	context->communicator = comm;

	ghost_mtraits_t crsTraits = {.format="CRS",.flags=GHOST_SPM_DEFAULT,.aux=NULL};
	CRSfullMatrix = ghost_initMatrix(&crsTraits);


	CRS_readHeader(CRSfullMatrix,matrixPath);  // read header

	if (ghost_getRank() == 0) {
		CRS_readRpt(CRSfullMatrix,matrixPath);  // read rpt
	}

	comm->wishes   = (ghost_mnnz_t*)       allocateMemory( nprocs*sizeof(ghost_mnnz_t), "comm->wishes" ); 
	comm->dues     = (ghost_mnnz_t*)       allocateMemory( nprocs*sizeof(ghost_mnnz_t), "comm->dues" ); 

	CRS_createDistribution(CRSfullMatrix,options,comm);

	DEBUG_LOG(1,"Mallocing space for %"PRmatIDX" rows",comm->lnrows[me]);

	if (ghost_getRank() != 0) {
		((CR_TYPE *)(CRSfullMatrix->data))->rpt = (ghost_midx_t *)malloc((comm->lnrows[me]+1)*sizeof(int));
	}

	if (ghost_getRank() == 0) {
		MPI_safecall(MPI_Scatterv(
					((CR_TYPE *)(CRSfullMatrix->data))->rpt, 
					(int *)comm->lnrows, 
					(int *)comm->lfRow, 
					MPI_INTEGER,
					MPI_IN_PLACE,
					(int)comm->lnrows[me],
					MPI_INTEGER, 0, MPI_COMM_WORLD));
	} else {
		MPI_safecall(MPI_Scatterv(
					((CR_TYPE *)(CRSfullMatrix->data))->rpt, 
					(int *)comm->lnrows, 
					(int *)comm->lfRow, 
					MPI_INTEGER,
					((CR_TYPE *)(CRSfullMatrix->data))->rpt,
					(int)comm->lnrows[me],
					MPI_INTEGER, 0, MPI_COMM_WORLD));
	}

	DEBUG_LOG(1,"Adjusting row pointers");

	for (i=0;i<comm->lnrows[me]+1;i++)
		((CR_TYPE *)(CRSfullMatrix->data))->rpt[i] =  ((CR_TYPE *)(CRSfullMatrix->data))->rpt[i] - comm->lfEnt[me]; 

	/* last entry of row_ptr holds the local number of entries */
	((CR_TYPE *)(CRSfullMatrix->data))->rpt[comm->lnrows[me]] = comm->lnEnts[me]; 

	DEBUG_LOG(1,"local rows          = %"PRmatIDX,comm->lnrows[me]);
	DEBUG_LOG(1,"local rows (offset) = %"PRmatIDX,comm->lfRow[me]);
	DEBUG_LOG(1,"local entries          = %"PRmatNNZ,comm->lnEnts[me]);
	DEBUG_LOG(1,"local entires (offset) = %"PRmatNNZ,comm->lfEnt[me]);

	CRS_readColValOffset(CRSfullMatrix,matrixPath, comm->lfEnt[me], comm->lfRow[me], comm->lnrows[me], comm->lnEnts[me], GHOST_IO_STD);

	DEBUG_LOG(1,"Adjust number of rows and number of nonzeros");
	((CR_TYPE *)(CRSfullMatrix->data))->nrows = comm->lnrows[me];
	((CR_TYPE *)(CRSfullMatrix->data))->nEnts = comm->lnEnts[me];

	CR_TYPE *locCR = NULL;
	CR_TYPE *remCR = NULL;

	CRS_createCommunication(CRSfullMatrix,&locCR,&remCR,options,context);

	context->fullMatrix = ghost_initMatrix(&traits[0]);
	context->fullMatrix->fromCRS(context->fullMatrix,CRSfullMatrix->data);

	context->localMatrix = ghost_initMatrix(&traits[1]);
	context->localMatrix->symmetry = CRSfullMatrix->symmetry;
	context->localMatrix->fromCRS(context->localMatrix,locCR);

	context->remoteMatrix = ghost_initMatrix(&traits[2]);
	context->remoteMatrix->fromCRS(context->remoteMatrix,remCR);

#ifdef OPENCL
		if (!(context->fullMatrix->traits->flags & GHOST_SPM_HOST))
			context->fullMatrix->CLupload(context->fullMatrix);
		if (!(context->localMatrix->traits->flags & GHOST_SPM_HOST))
			context->localMatrix->CLupload(context->localMatrix);
		if (!(context->remoteMatrix->traits->flags & GHOST_SPM_HOST))
			context->remoteMatrix->CLupload(context->remoteMatrix);
#endif
#ifdef CUDA
		if (!(context->fullMatrix->traits->flags & GHOST_SPM_HOST))
			context->fullMatrix->CUupload(context->fullMatrix);
		if (!(context->localMatrix->traits->flags & GHOST_SPM_HOST))
			context->localMatrix->CUupload(context->localMatrix);
		if (!(context->remoteMatrix->traits->flags & GHOST_SPM_HOST))
			context->remoteMatrix->CUupload(context->remoteMatrix);
#endif

	// TODO clean up
}


static void CRS_createCommunication(ghost_mat_t *mat, CR_TYPE **localCR, CR_TYPE **remoteCR, int options, ghost_context_t *context)
{
	CR_TYPE *fullCR = CR(mat);
	DEBUG_LOG(1,"Setting up communication");

	int hlpi;
	ghost_mnnz_t j;
	ghost_midx_t i;
	int me;
	ghost_mnnz_t max_loc_elements, thisentry;
	int *present_values;
	int acc_dues;
	int *tmp_transfers;
	int acc_wishes;
	int nEnts_glob;

	/* Counter how many entries are requested from each PE */
	int *item_from;

	ghost_mnnz_t *wishlist_counts;

	int *wishlist_mem,  **wishlist;
	int *cwishlist_mem, **cwishlist;

	int this_pseudo_col;
	int *pseudocol;
	int *globcol;
	int *revcol;

	int *comm_remotePE, *comm_remoteEl;
	ghost_mnnz_t lnEnts_l, lnEnts_r;
	int current_l, current_r;
	ghost_midx_t pseudo_ldim;
	int acc_transfer_wishes, acc_transfer_dues;

	size_t size_nint, size_col;
	size_t size_revc, size_a2ai, size_nptr, size_pval;  
	size_t size_mem, size_wish, size_dues;

	int nprocs = ghost_getNumberOfProcesses();
	ghost_comm_t *lcrp = context->communicator;

	size_nint = (size_t)( (size_t)(nprocs)   * sizeof(int)  );
	size_nptr = (size_t)( nprocs             * sizeof(int*) );
	size_a2ai = (size_t)( nprocs*nprocs * sizeof(int)  );

	me = ghost_getRank();

	max_loc_elements = 0;
	for (i=0;i<nprocs;i++)
		if (max_loc_elements<lcrp->lnrows[i]) max_loc_elements = lcrp->lnrows[i];

	nEnts_glob = lcrp->lfEnt[nprocs-1]+lcrp->lnEnts[nprocs-1]; 

	size_pval = (size_t)( max_loc_elements * sizeof(int) );
	size_revc = (size_t)( nEnts_glob       * sizeof(int) );
	size_col  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( int ) );


	item_from       = (int*) allocateMemory( size_nint, "item_from" ); 
	wishlist_counts = (ghost_mnnz_t *) allocateMemory( nprocs*sizeof(ghost_mnnz_t), "wishlist_counts" ); 
	comm_remotePE   = (int*) allocateMemory( size_col,  "comm_remotePE" );
	comm_remoteEl   = (int*) allocateMemory( size_col,  "comm_remoteEl" );
	present_values  = (int*) allocateMemory( size_pval, "present_values" ); 
	tmp_transfers   = (int*) allocateMemory( size_a2ai, "tmp_transfers" ); 
	pseudocol       = (int*) allocateMemory( size_col,  "pseudocol" );
	globcol         = (int*) allocateMemory( size_col,  "origincol" );
	revcol          = (int*) allocateMemory( size_revc, "revcol" );


	for (i=0; i<nprocs; i++) wishlist_counts[i] = 0;

	for (i=0;i<lcrp->lnEnts[me];i++){
		for (j=nprocs-1;j>=0; j--){
			if (lcrp->lfRow[j]<fullCR->col[i]+1) {
				comm_remotePE[i] = j;
				wishlist_counts[j]++;
				comm_remoteEl[i] = fullCR->col[i] -lcrp->lfRow[j];
				break;
			}
		}
	}

	acc_wishes = 0;
	for (i=0; i<nprocs; i++) acc_wishes += wishlist_counts[i];
	size_mem  = (size_t)( acc_wishes * sizeof(int) );

	wishlist        = (int**) allocateMemory( size_nptr, "wishlist" ); 
	cwishlist       = (int**) allocateMemory( size_nptr, "cwishlist" ); 
	wishlist_mem    = (int*)  allocateMemory( size_mem,  "wishlist_mem" ); 
	cwishlist_mem   = (int*)  allocateMemory( size_mem,  "cwishlist_mem" ); 

	hlpi = 0;
	for (i=0; i<nprocs; i++){
		wishlist[i]  = &wishlist_mem[hlpi];
		cwishlist[i] = &cwishlist_mem[hlpi];
		hlpi += wishlist_counts[i];
	}

	for (i=0;i<nprocs;i++) item_from[i] = 0;

	for (i=0;i<lcrp->lnEnts[me];i++){
		wishlist[comm_remotePE[i]][item_from[comm_remotePE[i]]] = comm_remoteEl[i];
		item_from[comm_remotePE[i]]++;
	}

	for (i=0; i<nprocs; i++){

		for (j=0; j<max_loc_elements; j++) present_values[j] = -1;

		if ( (i!=me) && (wishlist_counts[i]>0) ){
			thisentry = 0;
			for (j=0; j<wishlist_counts[i]; j++){
				if (present_values[wishlist[i][j]]<0){
					present_values[wishlist[i][j]] = thisentry;
					cwishlist[i][thisentry] = wishlist[i][j];
					thisentry = thisentry + 1;
				}
			}
			lcrp->wishes[i] = thisentry;
		}
		else lcrp->wishes[i] = 0; 

	}

	MPI_safecall(MPI_Allgather ( lcrp->wishes, nprocs, MPI_INTEGER, tmp_transfers, 
				nprocs, MPI_INTEGER, MPI_COMM_WORLD )) ;

	for (i=0; i<nprocs; i++) lcrp->dues[i] = tmp_transfers[i*nprocs+me];

	lcrp->dues[me] = 0; 

	acc_transfer_dues = 0;
	acc_transfer_wishes = 0;
	for (i=0; i<nprocs; i++){
		acc_transfer_wishes += lcrp->wishes[i];
		acc_transfer_dues   += lcrp->dues[i];
	}

	this_pseudo_col = lcrp->lnrows[me];
	lcrp->halo_elements = 0;
	for (i=0; i<nprocs; i++){
		if (i != me){ 
			for (j=0;j<lcrp->wishes[i];j++){
				pseudocol[lcrp->halo_elements] = this_pseudo_col;  
				globcol[lcrp->halo_elements]   = lcrp->lfRow[i]+cwishlist[i][j]; 
				revcol[globcol[lcrp->halo_elements]] = lcrp->halo_elements;
				lcrp->halo_elements++;
				this_pseudo_col++;
			}
		}
	}

	for (i=0;i<lcrp->lnEnts[me];i++)
	{
		if (comm_remotePE[i] == me) // local
			fullCR->col[i] =  comm_remoteEl[i];
		else // remote
			fullCR->col[i] = pseudocol[revcol[fullCR->col[i]]];
	}

	freeMemory ( size_col,  "comm_remoteEl",  comm_remoteEl);
	freeMemory ( size_col,  "comm_remotePE",  comm_remotePE);
	freeMemory ( size_col,  "pseudocol",      pseudocol);
	freeMemory ( size_col,  "globcol",        globcol);
	freeMemory ( size_revc, "revcol",         revcol);
	freeMemory ( size_pval, "present_values", present_values ); 

	size_wish = (size_t)( acc_transfer_wishes * sizeof(int) );
	size_dues = (size_t)( acc_transfer_dues   * sizeof(int) );

	MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	lcrp->wishlist      = (int**) allocateMemory( size_nptr, "lcrp->wishlist" ); 
	lcrp->duelist       = (int**) allocateMemory( size_nptr, "lcrp->duelist" ); 
	lcrp->wishlist_mem  = (int*)  allocateMemory( size_wish, "lcrp->wishlist_mem" ); 
	lcrp->duelist_mem   = (int*)  allocateMemory( size_dues, "lcrp->duelist_mem" ); 
	lcrp->wish_displ    = (int*)  allocateMemory( size_nint, "lcrp->wish_displ" ); 
	lcrp->due_displ     = (int*)  allocateMemory( size_nint, "lcrp->due_displ" ); 
	lcrp->hput_pos      = (int*)  allocateMemory( size_nptr, "lcrp->hput_pos" ); 

	acc_dues = 0;
	acc_wishes = 0;



	for (i=0; i<nprocs; i++){

		lcrp->due_displ[i]  = acc_dues;
		lcrp->wish_displ[i] = acc_wishes;
		lcrp->duelist[i]    = &(lcrp->duelist_mem[acc_dues]);
		lcrp->wishlist[i]   = &(lcrp->wishlist_mem[acc_wishes]);
		lcrp->hput_pos[i]   = lcrp->lnrows[me]+acc_wishes;

		if  ( (me != i) && !( (i == nprocs-2) && (me == nprocs-1) ) ){
			acc_dues   += lcrp->dues[i];
			acc_wishes += lcrp->wishes[i];
		}
	}

	for (i=0; i<nprocs; i++) for (j=0;j<lcrp->wishes[i];j++)
		lcrp->wishlist[i][j] = cwishlist[i][j]; 

	for(i=0; i<nprocs; i++) {
		MPI_safecall(MPI_Scatterv ( 
					lcrp->wishlist_mem, (int *)lcrp->wishes, lcrp->wish_displ, MPI_INTEGER, 
					lcrp->duelist[i], (int)lcrp->dues[i], MPI_INTEGER, i, MPI_COMM_WORLD ));
	}

	if (!(options & GHOST_OPTION_NO_SPLIT_SOLVERS)) { // split computation

		pseudo_ldim = lcrp->lnrows[me]+lcrp->halo_elements ;

		lnEnts_l=0;
		for (i=0; i<lcrp->lnEnts[me];i++) {
			if (fullCR->col[i]<lcrp->lnrows[me]) lnEnts_l++;
		}


		lnEnts_r = lcrp->lnEnts[me]-lnEnts_l;

		DEBUG_LOG(1,"PE%d: Rows=%"PRmatIDX"\t Ents=%"PRmatNNZ"(l),%"PRmatNNZ"(r),%"PRmatNNZ"(g)\t pdim=%"PRmatIDX, 
				me, lcrp->lnrows[me], lnEnts_l, lnEnts_r, lcrp->lnEnts[me], pseudo_ldim );

		//CR_TYPE *localCR;
		//CR_TYPE *remoteCR;


		(*localCR) = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");
		(*remoteCR) = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");

		(*localCR)->val = (ghost_mdat_t*) allocateMemory(lnEnts_l*sizeof( ghost_mdat_t ),"localMatrix->val" ); 
		(*localCR)->col = (ghost_midx_t*) allocateMemory(lnEnts_l*sizeof( ghost_midx_t ),"localMatrix->col" ); 
		(*localCR)->rpt = (ghost_midx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( ghost_midx_t ),"localMatrix->rpt" ); 

		(*remoteCR)->val = (ghost_mdat_t*) allocateMemory(lnEnts_r*sizeof( ghost_mdat_t ),"remoteMatrix->val" ); 
		(*remoteCR)->col = (ghost_midx_t*) allocateMemory(lnEnts_r*sizeof( ghost_midx_t ),"remoteMatrix->col" ); 
		(*remoteCR)->rpt = (ghost_midx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( ghost_midx_t ),"remoteMatrix->rpt" ); 

		//context->localMatrix->data = localCR;
		//context->remoteMatrix->data = remoteCR;

		(*localCR)->nrows = lcrp->lnrows[me];
		(*localCR)->nEnts = lnEnts_l;

		(*remoteCR)->nrows = lcrp->lnrows[me];
		(*remoteCR)->nEnts = lnEnts_r;

		//context->localMatrix->data = localCR;
		//context->localMatrix->nnz = lnEnts_l;
		//context->localMatrix->nrows = lcrp->lnrows[me];

		//context->remoteMatrix->data = remoteCR;
		//context->localMatrix->nnz = lnEnts_r;
		//context->localMatrix->nrows = lcrp->lnrows[me];

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) (*localCR)->val[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) (*localCR)->col[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) (*remoteCR)->val[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) (*remoteCR)->col[i] = 0.0;


		(*localCR)->rpt[0] = 0;
		(*remoteCR)->rpt[0] = 0;

		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
		DEBUG_LOG(1,"PE%d: lnrows=%"PRmatIDX" row_ptr=%"PRmatIDX"..%"PRmatIDX,
				me, lcrp->lnrows[me], fullCR->rpt[0], fullCR->rpt[lcrp->lnrows[me]]);
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

		for (i=0; i<lcrp->lnrows[me]; i++){

			current_l = 0;
			current_r = 0;

			for (j=fullCR->rpt[i]; j<fullCR->rpt[i+1]; j++){

				if (fullCR->col[j]<lcrp->lnrows[me]){
					(*localCR)->col[ (*localCR)->rpt[i]+current_l ] = fullCR->col[j]; 
					(*localCR)->val[ (*localCR)->rpt[i]+current_l ] = fullCR->val[j]; 
					current_l++;
				}
				else{
					(*remoteCR)->col[ (*remoteCR)->rpt[i]+current_r ] = fullCR->col[j];
					(*remoteCR)->val[ (*remoteCR)->rpt[i]+current_r ] = fullCR->val[j];
					current_r++;
				}

			}  

			(*localCR)->rpt[i+1] = (*localCR)->rpt[i] + current_l;
			(*remoteCR)->rpt[i+1] = (*remoteCR)->rpt[i] + current_r;
		}

		IF_DEBUG(3){
			for (i=0; i<lcrp->lnrows[me]+1; i++)
				DEBUG_LOG(3,"--Row_ptrs-- PE %d: i=%"PRmatIDX" local=%"PRmatIDX" remote=%"PRmatIDX, 
						me, i, (*localCR)->rpt[i], (*remoteCR)->rpt[i]);
			for (i=0; i<(*localCR)->rpt[lcrp->lnrows[me]]; i++)
				DEBUG_LOG(3,"-- local -- PE%d: localCR->col[%"PRmatIDX"]=%"PRmatIDX, me, i, (*localCR)->col[i]);
			for (i=0; i<(*remoteCR)->rpt[lcrp->lnrows[me]]; i++)
				DEBUG_LOG(3,"-- remote -- PE%d: remoteCR->col[%"PRmatIDX"]=%"PRmatIDX, me, i, (*remoteCR)->col[i]);
		}
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	}
	freeMemory ( size_mem,  "wishlist_mem",    wishlist_mem);
	freeMemory ( size_mem,  "cwishlist_mem",   cwishlist_mem);
	freeMemory ( size_nptr, "wishlist",        wishlist);
	freeMemory ( size_nptr, "cwishlist",       cwishlist);
	freeMemory ( size_a2ai, "tmp_transfers",   tmp_transfers);
	freeMemory ( size_nint, "wishlist_counts", wishlist_counts);
	freeMemory ( size_nint, "item_from",       item_from);




}

static void CRS_createDistribution(ghost_mat_t *mat, int options, ghost_comm_t *lcrp)
{
	CR_TYPE *cr = CR(mat);
	int me = ghost_getRank(); 
	ghost_mnnz_t j;
	ghost_midx_t i;
	int hlpi;
	int target_rows;
	int nprocs = ghost_getNumberOfProcesses();

	lcrp->lnEnts   = (ghost_mnnz_t*)       allocateMemory( nprocs*sizeof(ghost_mnnz_t), "lcrp->lnEnts" ); 
	lcrp->lnrows   = (ghost_midx_t*)       allocateMemory( nprocs*sizeof(ghost_midx_t), "lcrp->lnrows" ); 
	lcrp->lfEnt    = (ghost_mnnz_t*)       allocateMemory( nprocs*sizeof(ghost_mnnz_t), "lcrp->lfEnt" ); 
	lcrp->lfRow    = (ghost_midx_t*)       allocateMemory( nprocs*sizeof(ghost_midx_t), "lcrp->lfRow" ); 

	/****************************************************************************
	 *******  Calculate a fair partitioning of NZE and ROWS on master PE  *******
	 ***************************************************************************/
	if (me==0){

		if (options & GHOST_OPTION_WORKDIST_NZE){
			DEBUG_LOG(1,"Distribute Matrix with EQUAL_NZE on each PE");
			ghost_mnnz_t target_nnz;

			target_nnz = (cr->nEnts/nprocs)+1; /* sonst bleiben welche uebrig! */

			lcrp->lfRow[0]  = 0;
			lcrp->lfEnt[0] = 0;
			j = 1;

			for (i=0;i<cr->nrows;i++){
				if (cr->rpt[i] >= j*target_nnz){
					lcrp->lfRow[j] = i;
					lcrp->lfEnt[j] = cr->rpt[i];
					j = j+1;
				}
			}

		}
		else if (options & GHOST_OPTION_WORKDIST_LNZE){
			if (!(options & GHOST_OPTION_SERIAL_IO)) {
				DEBUG_LOG(0,"Warning! GHOST_OPTION_WORKDIST_LNZE has not (yet) been "
						"implemented for parallel IO! Switching to "
						"GHOST_OPTION_WORKDIST_NZE");
				options |= GHOST_OPTION_WORKDIST_NZE;
			} else {
				DEBUG_LOG(1,"Distribute Matrix with EQUAL_LNZE on each PE");
				ghost_mnnz_t *loc_count;
				int target_lnze;

				int trial_count, prev_count, trial_rows;
				int ideal, prev_rows;
				int outer_iter, outer_convergence;

				/* A first attempt should be blocks of equal size */
				target_rows = (cr->nrows/nprocs);

				lcrp->lfRow[0] = 0;
				lcrp->lfEnt[0] = 0;

				for (i=1; i<nprocs; i++){
					lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
					lcrp->lfEnt[i] = cr->rpt[lcrp->lfRow[i]];
				}

				for (i=0; i<nprocs-1; i++){
					lcrp->lnrows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
					lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
				}
				lcrp->lnrows[nprocs-1] = cr->nrows - lcrp->lfRow[nprocs-1] ;
				lcrp->lnEnts[nprocs-1] = cr->nEnts - lcrp->lfEnt[nprocs-1];

				/* Count number of local elements in each block */
				loc_count      = (ghost_mnnz_t*)       allocateMemory( nprocs*sizeof(ghost_mnnz_t), "loc_count" ); 
				for (i=0; i<nprocs; i++) loc_count[i] = 0;     

				for (i=0; i<nprocs; i++){
					for (j=lcrp->lfEnt[i]; j<lcrp->lfEnt[i]+lcrp->lnEnts[i]; j++){
						if (cr->col[j] >= lcrp->lfRow[i] && 
								cr->col[j]<lcrp->lfRow[i]+lcrp->lnrows[i])
							loc_count[i]++;
					}
				}
				DEBUG_LOG(2,"First run: local elements:");
				hlpi = 0;
				for (i=0; i<nprocs; i++){
					hlpi += loc_count[i];
					DEBUG_LOG(2,"Block %"PRmatIDX" %"PRmatNNZ" %"PRmatIDX, i, loc_count[i], lcrp->lnEnts[i]);
				}
				target_lnze = hlpi/nprocs;
				DEBUG_LOG(2,"total local elements: %d | per PE: %d", hlpi, target_lnze);

				outer_convergence = 0; 
				outer_iter = 0;

				while(outer_convergence==0){ 

					DEBUG_LOG(2,"Convergence Iteration %d", outer_iter);

					for (i=0; i<nprocs-1; i++){

						ideal = 0;
						prev_rows  = lcrp->lnrows[i];
						prev_count = loc_count[i];

						while (ideal==0){

							trial_rows = (int)( (double)(prev_rows) * sqrt((1.0*target_lnze)/(1.0*prev_count)) );

							/* Check ob die Anzahl der Elemente schon das beste ist das ich erwarten kann */
							if ( (trial_rows-prev_rows)*(trial_rows-prev_rows)<5.0 ) ideal=1;

							trial_count = 0;
							for (j=lcrp->lfEnt[i]; j<cr->rpt[lcrp->lfRow[i]+trial_rows]; j++){
								if (cr->col[j] >= lcrp->lfRow[i] && cr->col[j]<lcrp->lfRow[i]+trial_rows)
									trial_count++;
							}
							prev_rows  = trial_rows;
							prev_count = trial_count;
						}

						lcrp->lnrows[i]  = trial_rows;
						loc_count[i]     = trial_count;
						lcrp->lfRow[i+1] = lcrp->lfRow[i]+lcrp->lnrows[i];
						if (lcrp->lfRow[i+1]>cr->nrows) DEBUG_LOG(0,"Exceeded matrix dimension");
						lcrp->lfEnt[i+1] = cr->rpt[lcrp->lfRow[i+1]];
						lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;

					}

					lcrp->lnrows[nprocs-1] = cr->nrows - lcrp->lfRow[nprocs-1];
					lcrp->lnEnts[nprocs-1] = cr->nEnts - lcrp->lfEnt[nprocs-1];
					loc_count[nprocs-1] = 0;
					for (j=lcrp->lfEnt[nprocs-1]; j<cr->nEnts; j++)
						if (cr->col[j] >= lcrp->lfRow[nprocs-1]) loc_count[nprocs-1]++;

					DEBUG_LOG(2,"Next run: outer_iter=%d:", outer_iter);
					hlpi = 0;
					for (i=0; i<nprocs; i++){
						hlpi += loc_count[i];
						DEBUG_LOG(2,"Block %"PRmatIDX" %"PRmatNNZ" %"PRmatNNZ, i, loc_count[i], lcrp->lnEnts[i]);
					}
					target_lnze = hlpi/nprocs;
					DEBUG_LOG(2,"total local elements: %d | per PE: %d | total share: %6.3f%%",
							hlpi, target_lnze, 100.0*hlpi/(1.0*cr->nEnts));

					hlpi = 0;
					for (i=0; i<nprocs; i++) if ( (1.0*(loc_count[i]-target_lnze))/(1.0*target_lnze)>0.001) hlpi++;
					if (hlpi == 0) outer_convergence = 1;

					outer_iter++;

					if (outer_iter>20){
						DEBUG_LOG(0,"No convergence after 20 iterations, exiting iteration.");
						outer_convergence = 1;
					}

				}

				int p;
				for (p=0; i<nprocs; i++)  
					DEBUG_LOG(1,"PE%d lfRow=%"PRmatIDX" lfEnt=%"PRmatNNZ" lnrows=%"PRmatIDX" lnEnts=%"PRmatNNZ, p, lcrp->lfRow[i], lcrp->lfEnt[i], lcrp->lnrows[i], lcrp->lnEnts[i]);

				free(loc_count);
			}
		}
		else {

			DEBUG_LOG(1,"Distribute Matrix with EQUAL_ROWS on each PE");
			target_rows = (cr->nrows/nprocs);

			lcrp->lfRow[0] = 0;
			lcrp->lfEnt[0] = 0;

			for (i=1; i<nprocs; i++){
				lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
				lcrp->lfEnt[i] = cr->rpt[lcrp->lfRow[i]];
			}
		}


		for (i=0; i<nprocs-1; i++){
			lcrp->lnrows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
			lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
		}

		lcrp->lnrows[nprocs-1] = cr->nrows - lcrp->lfRow[nprocs-1] ;
		lcrp->lnEnts[nprocs-1] = cr->nEnts - lcrp->lfEnt[nprocs-1];
	}

	/****************************************************************************
	 *******            Distribute correct share to all PEs               *******
	 ***************************************************************************/

	MPI_safecall(MPI_Bcast(lcrp->lfRow,  nprocs, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lfEnt,  nprocs, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnrows, nprocs, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnEnts, nprocs, MPI_INTEGER, 0, MPI_COMM_WORLD));


}
#endif

static void CRS_readHeader(ghost_mat_t *mat, char *matrixPath)
{
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

static void CRS_readRpt(ghost_mat_t *mat, char *matrixPath)
{
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

static void CRS_readColValOffset(ghost_mat_t *mat, char *matrixPath, ghost_mnnz_t offsetEnts, ghost_midx_t offsetRows, ghost_midx_t nRows, ghost_mnnz_t nEnts, int IOtype)
{

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

/*int compareNZEPos( const void* a, const void* b ) 
{

	int aRow = ((NZE_TYPE*)a)->row,
		bRow = ((NZE_TYPE*)b)->row,
		aCol = ((NZE_TYPE*)a)->col,
		bCol = ((NZE_TYPE*)b)->col;

	if( aRow == bRow ) {
		return aCol - bCol;
	}
	else return aRow - bRow;
}*/

static void CRS_fromBin(ghost_mat_t *mat, char *matrixPath, ghost_context_t *ctx, int options)
{
	DEBUG_LOG(1,"Reading CRS matrix from file");

	if (ctx->flags & GHOST_CONTEXT_GLOBAL) {
		DEBUG_LOG(1,"Reading in a global context");
		CRS_readHeader(mat,matrixPath);
		CRS_readRpt(mat,matrixPath);
		CRS_readColValOffset(mat, matrixPath, 0, 0, CR(mat)->nrows, CR(mat)->nEnts, GHOST_IO_STD);
	} else {
#ifdef MPI
		DEBUG_LOG(1,"Reading in a distributed context");
		CRS_createDistributedContext(ctx,matrixPath,options,mat->traits);
#else
		UNUSED(options);
		ABORT("Trying to create a distributed context without MPI!");
#endif
	}
	DEBUG_LOG(1,"Matrix read in successfully");

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
	double hlp1;
	CR_TYPE *cr = CR(mat);

#pragma omp parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<cr->nrows; i++){
		hlp1 = 0.0;
		for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
			hlp1 = hlp1 + (double)cr->val[j] * rhsv[cr->col[j]];
	//		printf("%d: %d: %f*%f (%d) = %f\n",ghost_getRank(),i,cr->val[j],rhsv[cr->col[j]],cr->col[j],hlp1);
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
