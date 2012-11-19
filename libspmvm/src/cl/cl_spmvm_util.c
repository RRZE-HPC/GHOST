#include "spmvm_util.h"
#include "cl_matricks.h"
#include "cl_kernel.h"
#include "matricks.h"
#include "mpihelper.h"
#include <string.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <sys/param.h>

#define CL_MAX_DEVICE_NAME_LEN 500

static cl_command_queue queue;
static cl_context context;
static cl_program program;
static cl_kernel kernel[3];
static size_t globalSize[3];
static size_t globalSz;


/* -----------------------------------------------------------------------------
   Initiliaze OpenCL for the SpMVM, i.e., 
   - find platform for defined device type, 
   - select a device (there is only _ONE_ device per MPI process),
   - build program containing SpMVM kernels,
   - create SpMVM kernels
   -------------------------------------------------------------------------- */
void CL_init()
{
	cl_uint numPlatforms;
	cl_platform_id *platformIDs;
	cl_int err;
	unsigned int platform, device;
	char devicename[CL_MAX_DEVICE_NAME_LEN];
	int takedevice;
	int rank = getLocalRank();
	int size = getNumberOfRanksOnNode();
	char hostname[MAXHOSTNAMELEN];
	cl_uint numDevices;
	cl_device_id *deviceIDs;

	gethostname(hostname,MAXHOSTNAMELEN);


	CL_safecall(clGetPlatformIDs(0, NULL, &numPlatforms));
	platformIDs = (cl_platform_id *)allocateMemory(
			sizeof(cl_platform_id)*numPlatforms,"platformIDs");
	CL_safecall(clGetPlatformIDs(numPlatforms, platformIDs, NULL));

	for (platform=0; platform<numPlatforms; platform++) {
		CL_safecall(clGetDeviceIDs(platformIDs[platform],CL_MY_DEVICE_TYPE, 0, NULL,
					&numDevices));

		if (numDevices > 0) { // correct platform has been found
			break;
		}
	}

	deviceIDs = (cl_device_id *)allocateMemory(sizeof(cl_device_id)*numDevices,
			"deviceIDs");
	CL_safecall(clGetDeviceIDs(platformIDs[platform],CL_MY_DEVICE_TYPE, numDevices,
				deviceIDs, &numDevices));

	IF_DEBUG(1) {
		if ( 0 == rank ) {
			printf("## rank %i/%i on %s --\t Platform: %u, "
					"No. devices of desired type: %u\n", 
					rank, size-1, hostname, platform, numDevices);

			for( device = 0; device < numDevices; ++device) {
				CL_safecall(clGetDeviceInfo(deviceIDs[device],CL_DEVICE_NAME,
							sizeof(devicename),devicename,NULL));
				printf("## rank %i/%i on %s --\t Device %u: %s\n", 
						rank, size-1, hostname, device, devicename);
			}
		}
	}

	takedevice = rank%numDevices;
	CL_safecall(clGetDeviceInfo(deviceIDs[takedevice],CL_DEVICE_NAME,
				sizeof(devicename),devicename,NULL));
	IF_DEBUG(1) printf("## rank %i/%i on %s --\t Selecting device %d: %s\n", 
			rank, size-1,hostname, takedevice, devicename);

	size_t maxImgWidth;
	CL_safecall(clGetDeviceInfo(deviceIDs[takedevice],CL_DEVICE_IMAGE2D_MAX_WIDTH,
				sizeof(maxImgWidth),&maxImgWidth,NULL));
	IF_DEBUG(1) printf("## rank %i/%i on %s --\t Max image2d width: %lu\n", 
			rank, size-1,hostname, maxImgWidth);


	IF_DEBUG(1) printf("## rank %i/%i on %s --\t Creating context \n", rank,
			size-1, hostname);
	cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM,
		(cl_context_properties)platformIDs[platform],0};
	context = clCreateContext(cprops,1,&deviceIDs[takedevice],NULL,NULL,
			&err);
	CL_checkerror(err);


	IF_DEBUG(1) printf("## rank %i/%i on %s --\t Creating command queue\n", 
			rank, size-1, hostname);
	queue = clCreateCommandQueue(context,deviceIDs[takedevice],
			CL_QUEUE_PROFILING_ENABLE|CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
			&err);
	CL_checkerror(err);


#ifdef DOUBLE
#ifdef COMPLEX
	char opt[] = " -DDOUBLE -DCOMPLEX ";
#else
	char opt[] = " -DDOUBLE ";
#endif
#endif

#ifdef SINGLE
#ifdef COMPLEX
	char opt[] = " -DSINGLE -DCOMPLEX ";
#else
	char opt[] = " -DSINGLE ";
#endif
#endif

	program = CL_registerProgram("src/kernel.cl",opt);


	free(deviceIDs);
	free(platformIDs);
}

/* -----------------------------------------------------------------------------
   Create program inside previously created context (global variable) and 
   build it
   -------------------------------------------------------------------------- */
cl_program CL_registerProgram(char *filename, const char *opt)
{
	cl_program program;
	cl_int err;
	char *build_log;
	size_t log_size;
	cl_device_id deviceID;
	int size = getNumberOfRanksOnNode();
	int rank = getLocalRank();
	char hostname[MAXHOSTNAMELEN];

	gethostname(hostname,MAXHOSTNAMELEN);
	CL_safecall(clGetContextInfo(context,CL_CONTEXT_DEVICES,
				sizeof(cl_device_id),&deviceID,NULL));


	IF_DEBUG(1) printf("## rank %i/%i on %s --\t Creating program %s\n", rank, 
			size-1, hostname,basename(filename));
	program = clCreateProgramWithSource(context,1,(const char **)&kernelSource,
			NULL,&err);
	CL_checkerror(err);

	IF_DEBUG(1) printf("## rank %i/%i on %s --\t Building program with \"%s\""
			"and creating kernels\n", rank, size-1, hostname,opt);

	CL_safecall(clBuildProgram(program,1,&deviceID,opt,NULL,NULL));

	IF_DEBUG(1) {
		CL_safecall(clGetProgramBuildInfo(program,deviceID,
					CL_PROGRAM_BUILD_LOG,0,NULL,&log_size));
		build_log = (char *)allocateMemory(log_size+1,"build log");
		CL_safecall(clGetProgramBuildInfo(program,deviceID,
					CL_PROGRAM_BUILD_LOG,log_size,build_log,NULL));
		printf("Build log: %s",build_log);
	}

	return program;
}

cl_mem CL_allocDeviceMemoryMapped( size_t bytesize, void *hostPtr, int flag )
{
	cl_mem mem;
	cl_int err;

	mem = clCreateBuffer(context,flag|CL_MEM_USE_HOST_PTR,bytesize,
			hostPtr,&err);
	CL_checkerror(err);

	return mem;
}

cl_mem CL_allocDeviceMemory( size_t bytesize )
{
	if (bytesize == 0)
		return NULL;

	cl_mem mem;
	cl_int err;
	mem = clCreateBuffer(context,CL_MEM_READ_WRITE,bytesize,NULL,&err);

	CL_checkerror(err);

	return mem;
}

cl_mem CL_allocDeviceMemoryCached( size_t bytesize, void *hostPtr )
{
	cl_mem mem;
	cl_int err;
	cl_image_format image_format;

	image_format.image_channel_order = CL_RG;
	image_format.image_channel_mat_data_type = CL_FLOAT;

	mem = clCreateImage2D(context,CL_MEM_READ_WRITE,&image_format,bytesize/sizeof(mat_data_t),1,0,hostPtr,&err);

printf("image width: %lu\n",bytesize/sizeof(mat_data_t));	

	CL_checkerror(err);

	return mem;
}

void * CL_mapBuffer(cl_mem devmem, size_t bytesize)
{
	cl_int err;
	void * ret = clEnqueueMapBuffer(queue,devmem,CL_TRUE,CL_MAP_WRITE,0,
			bytesize,0,NULL,NULL,&err);
	CL_checkerror(err);
	return ret;
}

void CL_copyDeviceToHost(void* hostmem, cl_mem devmem, size_t bytesize) 
{
#ifdef CL_IMAGE
	const size_t origin[3] = {0,0,0};
	const size_t region[3] = {bytesize/sizeof(mat_data_t),0,0};
	CL_safecall(clEnqueueReadImage(queue,devmem,CL_TRUE,origin,region,0,0,
				hostmem,0,NULL,NULL));
#else
	int me;

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));
	IF_DEBUG(1) printf("PE%d: Copying back %lu data elements to host\n",me,bytesize/sizeof(mat_data_t));
	CL_safecall(clEnqueueReadBuffer(queue,devmem,CL_TRUE,0,bytesize,hostmem,0,
				NULL,NULL));
#endif
}

cl_event CL_copyDeviceToHostNonBlocking(void* hostmem, cl_mem devmem,
		size_t bytesize)
{
	cl_event event;
	CL_safecall(clEnqueueReadBuffer(queue,devmem,CL_FALSE,0,bytesize,hostmem,0,
				NULL,&event));
	return event;
}

void CL_copyHostToDeviceOffset(cl_mem devmem, void *hostmem,
		size_t bytesize, size_t offset)
{
	if (bytesize==0)
		return;
#ifdef CL_IMAGE
	cl_mem_object_type type;
	CL_safecall(clGetMemObjectInfo(devmem,CL_MEM_TYPE,sizeof(cl_mem_object_type),&type,NULL));
	if (type == CL_MEM_OBJECT_BUFFER) {
		CL_safecall(clEnqueueWriteBuffer(queue,devmem,CL_TRUE,offset,bytesize,
					hostmem,0,NULL,NULL));
	} else {
		const size_t origin[3] = {offset,0,0};
		const size_t region[3] = {bytesize/sizeof(mat_data_t),0,0};
		CL_safecall(clEnqueueWriteImage(queue,devmem,CL_TRUE,origin,region,0,0,
					hostmem,0,NULL,NULL));
	}
#else
	CL_safecall(clEnqueueWriteBuffer(queue,devmem,CL_TRUE,offset,bytesize,
				hostmem,0,NULL,NULL));
#endif
}

void CL_copyHostToDevice(cl_mem devmem, void *hostmem, size_t bytesize)
{
#ifdef CL_IMAGE
	cl_mem_object_type type;
	CL_safecall(clGetMemObjectInfo(devmem,CL_MEM_TYPE,sizeof(cl_mem_object_type),&type,NULL));
	if (type == CL_MEM_OBJECT_BUFFER) {
		CL_copyHostToDeviceOffset(devmem, hostmem, bytesize, 0);
	} else {
		const size_t origin[3] = {0,0,0};
		const size_t region[3] = {bytesize/sizeof(mat_data_t),0,0};
		CL_safecall(clEnqueueWriteImage(queue,devmem,CL_TRUE,origin,region,0,0,
					hostmem,0,NULL,NULL));
	}
#else
	CL_copyHostToDeviceOffset(devmem, hostmem, bytesize, 0);
#endif
}

void CL_freeDeviceMemory(cl_mem mem)
{
	if (mem)
		CL_safecall(clReleaseMemObject(mem));
}

void CL_bindMatrixToKernel(void *mat, int format, int T, int kernelIdx, int spmvmOptions) 
{
	cl_int err;

	if (mat == NULL)
		return;


	char kernelName[50] = "";
	strcat(kernelName, format==GHOST_SPM_GPUFORMAT_ELR?"ELR":"pJDS");
	char Tstr[2] = "";
	snprintf(Tstr,2,"%d",T);

	strcat(kernelName,Tstr);
	strcat(kernelName,"kernel");
	if (kernelIdx == GHOST_REMOTE_MAT_IDX || (spmvmOptions & GHOST_OPTION_AXPY))
		strcat(kernelName,"Add");


	kernel[kernelIdx] = clCreateKernel(program,kernelName,&err);

	CL_checkerror(err);

	if (format == GHOST_SPM_GPUFORMAT_ELR) {
		CL_ELR_TYPE *matrix = (CL_ELR_TYPE *)mat;
		globalSize[kernelIdx] = (size_t)matrix->padding*T;

		CL_safecall(clSetKernelArg(kernel[kernelIdx],2,sizeof(int),   
					&matrix->nrows));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],3,sizeof(int),   
					&matrix->padding));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],4,sizeof(cl_mem),
					&matrix->val));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],5,sizeof(cl_mem),
					&matrix->col));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],6,sizeof(cl_mem),
					&matrix->rowLen));
		globalSz = matrix->padding;
	} else {
		CL_PJDS_TYPE *matrix = (CL_PJDS_TYPE *)mat;
		globalSize[kernelIdx] = (size_t)matrix->padding*T;

		CL_safecall(clSetKernelArg(kernel[kernelIdx],2,sizeof(int),   
					&matrix->nrows));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],3,sizeof(cl_mem),
					&matrix->val));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],4,sizeof(cl_mem),
					&matrix->col));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],5,sizeof(cl_mem),
					&matrix->rowLen));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],6,sizeof(cl_mem),
					&matrix->colStart));
		globalSz = matrix->padding;
	}
	if (T>1) {
		CL_safecall(clSetKernelArg(kernel[kernelIdx],7,	sizeof(mat_data_t)*
					CL_getLocalSize(kernel[kernelIdx]),NULL));
	}
}

void CL_enqueueKernel(cl_kernel kernel)
{
	CL_safecall(clEnqueueNDRangeKernel(queue,kernel,1,NULL,&globalSz,NULL,0,NULL
				,NULL));
}

void CL_SpMVM(cl_mem rhsVec, cl_mem resVec, int type) 
{
	CL_safecall(clSetKernelArg(kernel[type],0,sizeof(cl_mem),&resVec));
	CL_safecall(clSetKernelArg(kernel[type],1,sizeof(cl_mem),&rhsVec));

	int me;

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));
	IF_DEBUG(1) printf("PE%d: Enqueueing SpMVM kernel with a global size of %lu\n",me,globalSize[type]);


	CL_safecall(clEnqueueNDRangeKernel(queue,kernel[type],1,NULL,
				&globalSize[type],NULL,0,NULL,NULL));
}

void CL_finish(int spmvmOptions) 
{

	if (!(spmvmOptions & GHOST_OPTION_NO_COMBINED_KERNELS)) {
		CL_safecall(clReleaseKernel(kernel[GHOST_FULL_MAT_IDX]));
	}
	if (!(spmvmOptions & GHOST_OPTION_NO_SPLIT_KERNELS)) {
		CL_safecall(clReleaseKernel(kernel[GHOST_LOCAL_MAT_IDX]));
		CL_safecall(clReleaseKernel(kernel[GHOST_REMOTE_MAT_IDX]));

	}

	CL_safecall(clReleaseCommandQueue(queue));
	CL_safecall(clReleaseContext(context));
}

void CL_uploadCRS(ghost_mat_t *matrix, GHOST_SPM_GPUFORMATS *matrixFormats, int spmvmOptions)
{
	
	if (!(matrix->format & GHOST_SPMFORMAT_DIST_CRS)) {
		DEBUG_LOG(0,"Device matrix can only be created from a distributed CRS host matrix.");
		return;
	}

	CL_createMatrix(matrix,matrixFormats,spmvmOptions);

	if (!(spmvmOptions & GHOST_OPTION_NO_COMBINED_KERNELS)) { // combined computation
		CL_bindMatrixToKernel(gpum->fullMatrix,gpum->fullFormat,
				matrixFormats->T[GHOST_FULL_MAT_IDX],GHOST_FULL_MAT_IDX, spmvmOptions);
	}

	if (!(spmvmOptions & GHOST_OPTION_NO_SPLIT_KERNELS)) { // split computation
		CL_bindMatrixToKernel(gpum->localMatrix,gpum->localFormat,
				matrixFormats->T[GHOST_LOCAL_MAT_IDX],GHOST_LOCAL_MAT_IDX, spmvmOptions);
		CL_bindMatrixToKernel(gpum->remoteMatrix,gpum->remoteFormat,
				matrixFormats->T[GHOST_REMOTE_MAT_IDX],GHOST_REMOTE_MAT_IDX, spmvmOptions);
	}

	return gpum;

}

void CL_createMatrix(ghost_mat_t* matrix, GHOST_SPM_GPUFORMATS *matrixFormats, int spmvmOptions)
{
	
	ghost_comm_t *lcrp = (ghost_comm_t *)matrix->matrix;
	GPUghost_mat_t * gpum = (GPUghost_mat_t*) allocateMemory( sizeof( GPUghost_mat_t ), "gpum" );

	int me = SpMVM_getRank();

	ELR_TYPE* elr 	= NULL;
	CL_ELR_TYPE* celr  = NULL;
	PJDS_TYPE* pjds	= NULL;
	CL_PJDS_TYPE* cpjds  = NULL;
	ELR_TYPE* lelr	= NULL;
	ELR_TYPE* relr	= NULL;
	CL_ELR_TYPE* lcelr = NULL;
	CL_ELR_TYPE* rcelr = NULL;
	PJDS_TYPE* lpjds= NULL;
	PJDS_TYPE* rpjds= NULL;
	CL_PJDS_TYPE* rcpjds= NULL;
	CL_PJDS_TYPE* lcpjds = NULL;


	DEBUG_LOG(1,"Creating device matrices");


	if (!(spmvmOptions & GHOST_OPTION_NO_COMBINED_KERNELS)) { // combined computation
		gpum->fullT = matrixFormats->T[GHOST_FULL_MAT_IDX];

		switch (matrixFormats->format[0]) {
			case GHOST_SPM_GPUFORMAT_PJDS:
				{
					DEBUG_LOG(1,"FULL pjds");

					pjds = CRStoPJDST( lcrp->val, lcrp->col, lcrp->lrow_ptr, 
							lcrp->lnrows[me],matrixFormats->T[0] );
					matrix->fullRowPerm = (int *)allocateMemory(
							sizeof(int)*lcrp->lnrows[me],"rowPerm");
					matrix->fullInvRowPerm = (int *)allocateMemory
						(sizeof(int)*lcrp->lnrows[me],"invRowPerm");
					memcpy(matrix->fullRowPerm, pjds->rowPerm,
							lcrp->lnrows[me]*sizeof(int));
					memcpy(matrix->fullInvRowPerm, pjds->invRowPerm,
							lcrp->lnrows[me]*sizeof(int));

					cpjds = CL_initPJDS( pjds );
					CL_uploadPJDS(cpjds, pjds);
					gpum->fullMatrix = cpjds;
					gpum->fullFormat = GHOST_SPM_GPUFORMAT_PJDS;

					freePJDS( pjds );
					break;
				}
			case GHOST_SPM_GPUFORMAT_ELR:
				{

					DEBUG_LOG(1,"FULL elr-%d", matrixFormats->T[0]);

					elr = CRStoELRT( lcrp->val, lcrp->col, lcrp->lrow_ptr, 
							lcrp->lnrows[me],matrixFormats->T[0] );
					celr = CL_initELR( elr );
					CL_uploadELR(celr, elr);
					gpum->fullMatrix = celr;
					gpum->fullFormat = GHOST_SPM_GPUFORMAT_ELR;

					freeELR( elr );
					break;
				}

		}

	}

	if (!(spmvmOptions & GHOST_OPTION_NO_SPLIT_KERNELS)) { // split computation
		gpum->localT = matrixFormats->T[GHOST_LOCAL_MAT_IDX];
		gpum->remoteT = matrixFormats->T[GHOST_REMOTE_MAT_IDX];

		if (matrixFormats->format[1] == GHOST_SPM_GPUFORMAT_PJDS && 
				matrixFormats->format[2] == GHOST_SPM_GPUFORMAT_PJDS)
			ABORT("The matrix format must _not_ be pJDS for the "
					"local and remote part of the matrix.");

		if (matrixFormats->format[1] == GHOST_SPM_GPUFORMAT_PJDS) {
			DEBUG_LOG(1,"LOCAL pjds");

			lpjds = CRStoPJDST( lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l, 
					lcrp->lnrows[me],matrixFormats->T[1] );

			// allocate space for permutations in lcrp
			matrix->splitRowPerm = (int *)allocateMemory(
					sizeof(int)*lcrp->lnrows[me],"rowPerm");
			matrix->splitInvRowPerm = (int *)allocateMemory(
					sizeof(int)*lcrp->lnrows[me],"invRowPerm");

			// save permutations from matrix to lcrp
			memcpy(matrix->splitRowPerm, lpjds->rowPerm, 
					lcrp->lnrows[me]*sizeof(int));
			memcpy(matrix->splitInvRowPerm, lpjds->invRowPerm, 
					lcrp->lnrows[me]*sizeof(int));

			lcpjds = CL_initPJDS( lpjds );
			CL_uploadPJDS(lcpjds, lpjds);
			gpum->localMatrix = lcpjds;
			gpum->localFormat = GHOST_SPM_GPUFORMAT_PJDS;

			freePJDS( lpjds );
		}
		if (matrixFormats->format[2] == GHOST_SPM_GPUFORMAT_PJDS) {
			DEBUG_LOG(1,"REMOTE pjds");

			rpjds = CRStoPJDST( lcrp->rval, lcrp->rcol, lcrp->lrow_ptr_r, 
					lcrp->lnrows[me],matrixFormats->T[2] );

			matrix->splitRowPerm = (int *)allocateMemory(
					sizeof(int)*lcrp->lnrows[me],"rowPerm");
			matrix->splitInvRowPerm = (int *)allocateMemory(
					sizeof(int)*lcrp->lnrows[me],"invRowPerm");

			memcpy(matrix->splitRowPerm, rpjds->rowPerm,
					lcrp->lnrows[me]*sizeof(int));
			memcpy(matrix->splitInvRowPerm, rpjds->invRowPerm,
					lcrp->lnrows[me]*sizeof(int));


			rcpjds = CL_initPJDS( rpjds );
			CL_uploadPJDS(rcpjds, rpjds);
			gpum->remoteMatrix = rcpjds;
			gpum->remoteFormat = GHOST_SPM_GPUFORMAT_PJDS;


			freePJDS( rpjds );


		}
		if (matrixFormats->format[1] == GHOST_SPM_GPUFORMAT_ELR) {
			DEBUG_LOG(1,"LOCAL elr");

			if (matrixFormats->format[2] == GHOST_SPM_GPUFORMAT_PJDS) { // remote pJDS
				lelr = CRStoELRTP(lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l,
						lcrp->lnrows[me],gpum->splitInvRowPerm,
						matrixFormats->T[1]);
			} else { // remote ELR
				lelr = CRStoELRT(lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l,
						lcrp->lnrows[me],matrixFormats->T[1]);
			}

			lcelr = CL_initELR( lelr );
			CL_uploadELR(lcelr, lelr);
			gpum->localMatrix = lcelr;
			gpum->localFormat = GHOST_SPM_GPUFORMAT_ELR;

			freeELR( lelr ); // FIXME run failes for some configurations if enabled (or not?)
		}
		if (matrixFormats->format[2] == GHOST_SPM_GPUFORMAT_ELR) {
			DEBUG_LOG(1,"REMOTE elr");

			if (matrixFormats->format[1] == GHOST_SPM_GPUFORMAT_PJDS) { // local pJDS
				relr = CRStoELRTP(lcrp->rval, lcrp->rcol, lcrp->lrow_ptr_r,
						lcrp->lnrows[me],gpum->splitInvRowPerm,
						matrixFormats->T[2]);
			} else { // local ELR
				relr = CRStoELRT( lcrp->rval, lcrp->rcol, lcrp->lrow_ptr_r,
						lcrp->lnrows[me],matrixFormats->T[2] );
			}


			rcelr = CL_initELR( relr );
			CL_uploadELR(rcelr, relr);
			gpum->remoteMatrix = rcelr;
			gpum->remoteFormat = GHOST_SPM_GPUFORMAT_ELR;

			freeELR( relr ); // FIXME run failes for some configurations if enabled (or not?)
		}
	}
	matrix->devMatrix =  gpum;
}

void CL_uploadVector( ghost_vec_t *vec )
{
	CL_copyHostToDevice(vec->CL_val_gpu,vec->val,vec->nrows*sizeof(mat_data_t));
}

void CL_downloadVector( ghost_vec_t *vec )
{
	CL_copyDeviceToHost(vec->val,vec->CL_val_gpu,vec->nrows*sizeof(mat_data_t));
}

size_t CL_getLocalSize(cl_kernel kernel) 
{

	cl_device_id deviceID;
	size_t wgSize;

	CL_safecall(clGetContextInfo(context,CL_CONTEXT_DEVICES,
				sizeof(cl_device_id),&deviceID,NULL));
	CL_safecall(clGetKernelWorkGroupInfo(kernel,deviceID,
				CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t),&wgSize,NULL));

	return wgSize;
}

static int stringcmp(const void *x, const void *y)
{
	return (strcmp((char *)x, (char *)y));
}

CL_DEVICE_INFO *CL_getDeviceInfo() 
{
	CL_DEVICE_INFO *devInfo = allocateMemory(sizeof(CL_DEVICE_INFO),"devInfo");
	devInfo->nDistinctDevices = 1;

	int me,size,i;
	cl_device_id deviceID;
	char name[CL_MAX_DEVICE_NAME_LEN];
	char *names = NULL;

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&me));
	MPI_safecall(MPI_Comm_size(MPI_COMM_WORLD,&size));


	CL_safecall(clGetContextInfo(context,CL_CONTEXT_DEVICES,
				sizeof(cl_device_id),&deviceID,NULL));
	CL_safecall(clGetDeviceInfo(deviceID,CL_DEVICE_NAME,
				CL_MAX_DEVICE_NAME_LEN*sizeof(char),name,NULL));

	if (me==0) {
		names = (char *)allocateMemory(size*CL_MAX_DEVICE_NAME_LEN*sizeof(char),
				"names");
	}


	MPI_safecall(MPI_Gather(name,CL_MAX_DEVICE_NAME_LEN,MPI_CHAR,names,
				CL_MAX_DEVICE_NAME_LEN,MPI_CHAR,0,MPI_COMM_WORLD));

	if (me==0) {
		qsort(names,size,CL_MAX_DEVICE_NAME_LEN*sizeof(char),stringcmp);
		for (i=1; i<size; i++) {
			if (strcmp(names+(i-1)*CL_MAX_DEVICE_NAME_LEN,
						names+i*CL_MAX_DEVICE_NAME_LEN)) {
				devInfo->nDistinctDevices++;
			}
		}
	}

	MPI_safecall(MPI_Bcast(&devInfo->nDistinctDevices,1,MPI_INT,0,MPI_COMM_WORLD));

	devInfo->nDevices = allocateMemory(sizeof(int)*devInfo->nDistinctDevices,"nDevices");
	devInfo->names = allocateMemory(sizeof(char *)*devInfo->nDistinctDevices,"device names");
	for (i=0; i<devInfo->nDistinctDevices; i++)
		devInfo->names[i] = allocateMemory(sizeof(char)*CL_MAX_DEVICE_NAME_LEN,"device names");

	if (me==0) {
		strncpy(devInfo->names[0],names,CL_MAX_DEVICE_NAME_LEN);
		devInfo->nDevices[0] = 1;


		int distIdx = 0;
		for (i=1; i<size; i++) {
			devInfo->nDevices[distIdx]++;
			if (strcmp(names+(i-1)*CL_MAX_DEVICE_NAME_LEN,
						names+i*CL_MAX_DEVICE_NAME_LEN)) {
				strncpy(devInfo->names[distIdx],names+i*CL_MAX_DEVICE_NAME_LEN,CL_MAX_DEVICE_NAME_LEN);
				distIdx++;
			}
		}

		free(names);
	}


	MPI_safecall(MPI_Bcast(devInfo->nDevices,devInfo->nDistinctDevices,MPI_INT,0,MPI_COMM_WORLD));

	for (i=0; i<devInfo->nDistinctDevices; i++)
		MPI_safecall(MPI_Bcast(devInfo->names[i],CL_MAX_DEVICE_NAME_LEN,MPI_CHAR,0,MPI_COMM_WORLD));


	return devInfo;
}

void destroyCLdeviceInfo(CL_DEVICE_INFO * di) 
{

	if (di) {	
		int i;
		for (i=0; i<di->nDistinctDevices; i++) {
			free(di->names[i]);
		}
		free(di->names);
		free(di->nDevices);
		free(di);
	}

}
