#include "oclfun.h"
#include "oclmacros.h"
#include "matricks.h"
#include <string.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>

#define CL_MAX_DEVICE_NAME_LEN 500

static cl_command_queue queue;
static cl_context context;
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
void CL_init(SPM_GPUFORMATS *matFormats)
{
	cl_uint numPlatforms;
	cl_platform_id *platformIDs;
	cl_int err;
	cl_program program;
	unsigned int platform, device;
	char devicename[CL_MAX_DEVICE_NAME_LEN];
	int takedevice;
	int rank;
	int size;
	char hostname[MAXHOSTNAMELEN];
	cl_uint numDevices;
	cl_device_id *deviceIDs;

	gethostname(hostname,MAXHOSTNAMELEN);
	MPI_safecall(MPI_Comm_size( single_node_comm, &size));
	MPI_safecall(MPI_Comm_rank( single_node_comm, &rank));


	CL_safecall(clGetPlatformIDs(0, NULL, &numPlatforms));
	platformIDs = (cl_platform_id *)allocateMemory(
			sizeof(cl_platform_id)*numPlatforms,"platformIDs");
	CL_safecall(clGetPlatformIDs(numPlatforms, platformIDs, NULL));

	for (platform=0; platform<numPlatforms; platform++) {
		CL_safecall(clGetDeviceIDs(platformIDs[platform],CL_DEVTYPE, 0, NULL,
					&numDevices));

		if (numDevices > 0) { // correct platform has been found
			break;
		}
	}

	deviceIDs = (cl_device_id *)allocateMemory(sizeof(cl_device_id)*numDevices,
			"deviceIDs");
	CL_safecall(clGetDeviceIDs(platformIDs[platform],CL_DEVTYPE, numDevices,
				deviceIDs, &numDevices));

	IF_DEBUG(1) {
		if ( 0 == rank ) {
			printf("## rank %i/%i on %s --\t Platform: %u, \
					No. devices of desired type: %u\n", 
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
	char *opt = " -DDOUBLE -DCOMPLEX ";
#else
	char *opt = " -DDOUBLE ";
#endif
#endif

#ifdef SINGLE
#ifdef COMPLEX
	char *opt = " -DSINGLE -DCOMPLEX ";
#else
	char *opt = " -DSINGLE ";
#endif
#endif

	program = CL_registerProgram("src/kernel.cl",opt);

	int i;
	for (i=0; i<3; i++) {

		char kernelName[50] = "";
		strcat(kernelName, 
				matFormats->format[i]==SPM_GPUFORMAT_ELR?"ELR":"pJDS");
		char Tstr[2] = "";
		snprintf(Tstr,2,"%d",matFormats->T[i]);

		strcat(kernelName,Tstr);
		strcat(kernelName,"kernel");
		if (i==SPM_KERNEL_REMOTE || (SPMVM_OPTIONS & SPMVM_OPTION_AXPY))
			strcat(kernelName,"Add");


		kernel[i] = clCreateKernel(program,kernelName,&err);

		IF_DEBUG(1) printf("creating kernel %s\n",kernelName);
		CL_checkerror(err);
	}

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
	FILE *fp;
	char *source_str;
	size_t source_size;
	char *build_log;
	size_t log_size;
	long filesize;
	cl_device_id deviceID;
	int size, rank;
	char hostname[MAXHOSTNAMELEN];

	gethostname(hostname,MAXHOSTNAMELEN);
	MPI_safecall(MPI_Comm_size( single_node_comm, &size));
	MPI_safecall(MPI_Comm_rank( single_node_comm, &rank));
	CL_safecall(clGetContextInfo(context,CL_CONTEXT_DEVICES,
				sizeof(cl_device_id),&deviceID,NULL));

	fp = fopen(filename, "r");
	if (!fp) {
		char cerr[] = "Failed to load kernel file: ";
		char msg[strlen(cerr)+strlen(filename)];
		strcpy(cerr,msg);
		strcat(msg,filename);
		myabort(msg);
	}

	fseek(fp,0L,SEEK_END);
	filesize = ftell(fp);
	fseek(fp,0L,SEEK_SET);

	source_str = (char*)allocateMemory(filesize,"source");
	source_size = fread( source_str, 1, filesize, fp);
	fclose( fp );

	IF_DEBUG(1) printf("## rank %i/%i on %s --\t Creating program %s\n", rank, 
			size-1, hostname,basename(filename));

	program = clCreateProgramWithSource(context,1,(const char **)&source_str,
			&source_size,&err);
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

cl_mem CL_allocDeviceMemoryMapped( size_t bytesize, void *hostPtr )
{
	cl_mem mem;
	cl_int err;

	mem = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,bytesize,
			hostPtr,&err);
	CL_checkerror(err);

	return mem;
}

cl_mem CL_allocDeviceMemory( size_t bytesize )
{
	cl_mem mem;
	cl_int err;
	mem = clCreateBuffer(context,CL_MEM_READ_WRITE,bytesize,NULL,&err);

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
	CL_safecall(clEnqueueReadBuffer(queue,devmem,CL_TRUE,0,bytesize,hostmem,0,
				NULL,NULL));
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
	CL_safecall(clEnqueueWriteBuffer(queue,devmem,CL_TRUE,offset,bytesize,
				hostmem,0,NULL,NULL));
}

void CL_copyHostToDevice(cl_mem devmem, void *hostmem, size_t bytesize)
{
	CL_copyHostToDeviceOffset(devmem, hostmem, bytesize, 0);
}

void CL_freeDeviceMemory(cl_mem mem)
{
	CL_safecall(clReleaseMemObject(mem));
}

void CL_bindMatrixToKernel(void *mat, int format, int T, int kernelIdx) 
{
	if (mat == NULL)
		return;

	if (format == SPM_GPUFORMAT_ELR) {
		CL_ELR_TYPE *matrix = (CL_ELR_TYPE *)mat;
		globalSize[kernelIdx] = (size_t)matrix->padding*T;

		CL_safecall(clSetKernelArg(kernel[kernelIdx],2,sizeof(int),   
					&matrix->nRows));
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
					&matrix->nRows));
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
		CL_safecall(clSetKernelArg(kernel[kernelIdx],7,	sizeof(real)*
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


	CL_safecall(clEnqueueNDRangeKernel(queue,kernel[type],1,NULL,
				&globalSize[type],NULL,0,NULL,NULL));
}

void CL_finish() 
{

	int i;

	for (i=0; i<3; i++) 
		CL_safecall(clReleaseKernel(kernel[i]));

	CL_safecall(clReleaseCommandQueue(queue));
	CL_safecall(clReleaseContext(context));


}

void CL_uploadCRS(LCRP_TYPE *lcrp, SPM_GPUFORMATS *matrixFormats)
{

	CL_init(matrixFormats);
	CL_setup_communication(lcrp,matrixFormats);

	if (SPMVM_KERNELS_SELECTED & SPMVM_KERNELS_COMBINED) {
		CL_bindMatrixToKernel(lcrp->fullMatrix,lcrp->fullFormat,
				matrixFormats->T[SPM_KERNEL_FULL],SPM_KERNEL_FULL);
	}

	if (SPMVM_KERNELS_SELECTED & SPMVM_KERNELS_SPLIT) {
		CL_bindMatrixToKernel(lcrp->localMatrix,lcrp->localFormat,
				matrixFormats->T[SPM_KERNEL_LOCAL],SPM_KERNEL_LOCAL);
		CL_bindMatrixToKernel(lcrp->remoteMatrix,lcrp->remoteFormat,
				matrixFormats->T[SPM_KERNEL_REMOTE],SPM_KERNEL_REMOTE);
	}


}

void CL_setup_communication(LCRP_TYPE* lcrp, SPM_GPUFORMATS *matrixFormats)
{

	ELR_TYPE* elr 	= NULL;
	ELR_TYPE* lelr	= NULL;
	ELR_TYPE* relr	= NULL;
	CL_ELR_TYPE* celr  = NULL;
	CL_ELR_TYPE* lcelr = NULL;
	CL_ELR_TYPE* rcelr = NULL;
	PJDS_TYPE* pjds	= NULL;
	PJDS_TYPE* lpjds= NULL;
	PJDS_TYPE* rpjds= NULL;
	CL_PJDS_TYPE* rcpjds= NULL;
	CL_PJDS_TYPE* cpjds  = NULL;
	CL_PJDS_TYPE* lcpjds = NULL;

	int me;

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));
	IF_DEBUG(1) printf("PE%i: creating matrices:\n", me);


	if (SPMVM_KERNELS_SELECTED & SPMVM_KERNELS_COMBINED) { // combined computation
		switch (matrixFormats->format[0]) {
			case SPM_GPUFORMAT_PJDS:
				{
					IF_DEBUG(1) printf("PE%i: FULL pjds:\n", me);

					pjds = CRStoPJDST( lcrp->val, lcrp->col, lcrp->lrow_ptr, 
							lcrp->lnRows[me],matrixFormats->T[0] );
					lcrp->fullRowPerm = (int *)allocateMemory(
							sizeof(int)*lcrp->lnRows[me],"rowPerm");
					lcrp->fullInvRowPerm = (int *)allocateMemory
						(sizeof(int)*lcrp->lnRows[me],"invRowPerm");
					memcpy(lcrp->fullRowPerm, pjds->rowPerm,
							lcrp->lnRows[me]*sizeof(int));
					memcpy(lcrp->fullInvRowPerm, pjds->invRowPerm,
							lcrp->lnRows[me]*sizeof(int));

					cpjds = CL_initPJDS( pjds );
					CL_uploadPJDS(cpjds, pjds);
					lcrp->fullMatrix = cpjds;
					lcrp->fullFormat = SPM_GPUFORMAT_PJDS;

					freePJDS( pjds );
					break;
				}
			case SPM_GPUFORMAT_ELR:
				{

					IF_DEBUG(1) printf("PE%i: FULL elr-%d:\n", me,
							matrixFormats->T[0]);

					elr = CRStoELRT( lcrp->val, lcrp->col, lcrp->lrow_ptr, 
							lcrp->lnRows[me],matrixFormats->T[0] );
					celr = CL_initELR( elr );
					CL_uploadELR(celr, elr);
					lcrp->fullMatrix = celr;
					lcrp->fullFormat = SPM_GPUFORMAT_ELR;

					freeELR( elr );
					break;
				}

		}

	}

	if (SPMVM_KERNELS_SELECTED & SPMVM_KERNELS_SPLIT) { // split computation

		if (matrixFormats->format[1] == SPM_GPUFORMAT_PJDS && 
				matrixFormats->format[2] == SPM_GPUFORMAT_PJDS)
			myabort("The matrix format must _not_ be pJDS for the"
					"local and remote part of the matrix.");

		if (matrixFormats->format[1] == SPM_GPUFORMAT_PJDS) {
			IF_DEBUG(1) printf("PE%i: LOCAL pjds:\n", me);

			lpjds = CRStoPJDST( lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l, 
					lcrp->lnRows[me],matrixFormats->T[1] );

			// allocate space for permutations in lcrp
			lcrp->splitRowPerm = (int *)allocateMemory(
					sizeof(int)*lcrp->lnRows[me],"rowPerm");
			lcrp->splitInvRowPerm = (int *)allocateMemory(
					sizeof(int)*lcrp->lnRows[me],"invRowPerm");

			// save permutations from matrix to lcrp
			memcpy(lcrp->splitRowPerm, lpjds->rowPerm, 
					lcrp->lnRows[me]*sizeof(int));
			memcpy(lcrp->splitInvRowPerm, lpjds->invRowPerm, 
					lcrp->lnRows[me]*sizeof(int));

			lcpjds = CL_initPJDS( lpjds );
			CL_uploadPJDS(lcpjds, lpjds);
			lcrp->localMatrix = lcpjds;
			lcrp->localFormat = SPM_GPUFORMAT_PJDS;

			freePJDS( lpjds );
		}
		if (matrixFormats->format[2] == SPM_GPUFORMAT_PJDS) {
			IF_DEBUG(1) printf("PE%i: REMOTE pjds:\n", me);

			rpjds = CRStoPJDST( lcrp->rval, lcrp->rcol, lcrp->lrow_ptr_r, 
					lcrp->lnRows[me],matrixFormats->T[2] );

			lcrp->splitRowPerm = (int *)allocateMemory(
					sizeof(int)*lcrp->lnRows[me],"rowPerm");
			lcrp->splitInvRowPerm = (int *)allocateMemory(
					sizeof(int)*lcrp->lnRows[me],"invRowPerm");

			memcpy(lcrp->splitRowPerm, rpjds->rowPerm,
					lcrp->lnRows[me]*sizeof(int));
			memcpy(lcrp->splitInvRowPerm, rpjds->invRowPerm,
					lcrp->lnRows[me]*sizeof(int));


			rcpjds = CL_initPJDS( rpjds );
			CL_uploadPJDS(rcpjds, rpjds);
			lcrp->remoteMatrix = rcpjds;
			lcrp->remoteFormat = SPM_GPUFORMAT_PJDS;

			freePJDS( rpjds );


		}
		if (matrixFormats->format[1] == SPM_GPUFORMAT_ELR) {
			IF_DEBUG(1) printf("PE%i: LOCAL elr:\n", me);

			if (matrixFormats->format[2] == SPM_GPUFORMAT_PJDS) { // remote pJDS
				lelr = CRStoELRTP(lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l,
						lcrp->lnRows[me],lcrp->splitInvRowPerm,
						matrixFormats->T[1]);
			} else { // remote ELR
				lelr = CRStoELRT(lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l,
						lcrp->lnRows[me],matrixFormats->T[1]);
			}

			lcelr = CL_initELR( lelr );
			CL_uploadELR(lcelr, lelr);
			lcrp->localMatrix = lcelr;
			lcrp->localFormat = SPM_GPUFORMAT_ELR;

			freeELR( lelr );
		}
		if (matrixFormats->format[2] == SPM_GPUFORMAT_ELR) {
			IF_DEBUG(1) printf("PE%i: REMOTE elr:\n", me);

			if (matrixFormats->format[1] == SPM_GPUFORMAT_PJDS) { // local pJDS
				relr = CRStoELRTP(lcrp->rval, lcrp->rcol, lcrp->lrow_ptr_r,
						lcrp->lnRows[me],lcrp->splitInvRowPerm,
						matrixFormats->T[2]);
			} else { // local ELR
				relr = CRStoELRT( lcrp->rval, lcrp->rcol, lcrp->lrow_ptr_r,
						lcrp->lnRows[me],matrixFormats->T[2] );
			}


			rcelr = CL_initELR( relr );
			CL_uploadELR(rcelr, relr);
			lcrp->remoteMatrix = rcelr;
			lcrp->remoteFormat = SPM_GPUFORMAT_ELR;

			freeELR( relr ); 
		}
	}
}

void CL_uploadVector( VECTOR_TYPE *vec )
{
	CL_copyHostToDevice(vec->CL_val_gpu,vec->val,vec->nRows*sizeof(real));
}

void CL_downloadVector( VECTOR_TYPE *vec )
{
	CL_copyDeviceToHost(vec->val,vec->CL_val_gpu,vec->nRows*sizeof(real));
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
	char *names;

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
