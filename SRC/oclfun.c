#include "oclfun.h"
#include "oclmacros.h"
#include "matricks.h"
#include <string.h>

extern int SPMVM_OPTIONS;

static cl_command_queue queue;
static cl_context context;
static cl_kernel kernel[6];
static size_t localSize[6] = {256,256,256,256,256,256};
static size_t globalSize[6];

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
	fprintf(stderr,"OpenCL error (via pfn_notify): %s\n",errinfo);
}

void CL_init( int rank, int size, const char* hostname, MATRIX_FORMATS *matrixFormats) {
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_platform_id *platformIDs;
	cl_device_id *deviceIDs;
	cl_int err;
	cl_program program[3];
	int platform, device, takedevice;
	char devicename[1024];



	cl_uint dummy;

	CL_safecall(clGetPlatformIDs(0, NULL, &numPlatforms));
	platformIDs = (cl_platform_id *)allocateMemory(sizeof(cl_platform_id)*numPlatforms,"platformIDs"); // segfault with aligned mem


	CL_safecall(clGetPlatformIDs(numPlatforms, platformIDs, &dummy));



	for (platform=0; platform<numPlatforms; platform++) {
		CL_safecall(clGetDeviceIDs(platformIDs[platform],CL_DEVTYPE, 0, NULL, &numDevices));

		if (numDevices > 0) { // correct platform has been found
			break;
		}

	}
	deviceIDs = (cl_device_id *)allocateMemory(sizeof(cl_device_id)*numDevices,"deviceIDs");
	CL_safecall(clGetDeviceIDs(platformIDs[platform],CL_DEVTYPE, numDevices, deviceIDs, &numDevices));

	if ( 0 == rank ) {
		IF_DEBUG(1) printf("## rank %i/%i on %s --\t Platform: %d, No. devices of desired type: %d\n", 
				rank, size-1, hostname, platform, numDevices);

		for( device = 0; device < numDevices; ++device) {
			CL_safecall(clGetDeviceInfo(deviceIDs[device],CL_DEVICE_NAME,sizeof(devicename),devicename,NULL));
			IF_DEBUG(1) printf("## rank %i/%i on %s --\t Device %d: %s\n", 
					rank, size-1, hostname, device, devicename);
		}
	}

	takedevice = rank%numDevices;
	CL_safecall(clGetDeviceInfo(deviceIDs[takedevice],CL_DEVICE_NAME,sizeof(devicename),devicename,NULL));
	printf("## rank %i/%i on %s --\t Selecting device %d: %s\n", rank, size-1, hostname, takedevice, devicename);

	IF_DEBUG(1) printf("## rank %i/%i on %s --\t Creating context \n", rank, size-1, hostname);
	cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[platform],0};
	context = clCreateContext(cprops,1,&deviceIDs[takedevice],pfn_notify,NULL,&err);
	CL_checkerror(err);

	IF_DEBUG(1) printf("## rank %i/%i on %s --\t Creating command queue\n", rank, size-1, hostname);
	queue = clCreateCommandQueue(context,deviceIDs[takedevice],CL_QUEUE_PROFILING_ENABLE|CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&err);
	CL_checkerror(err);


	FILE *fp;
	char *source_str;
	size_t source_size;
	char *build_log;
	size_t log_size;


	fp = fopen("SRC/kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)allocateMemory(10000,"source");
	source_size = fread( source_str, 1, 10000, fp);
	fclose( fp );



	int i;
	for (i=0; i<3; i++) {
		IF_DEBUG(1) printf("## rank %i/%i on %s --\t Creating program\n", rank, size-1, hostname);
		program[i] = clCreateProgramWithSource(context,1,(const char **)&source_str,&source_size,&err);
		CL_checkerror(err);


		IF_DEBUG(1) printf("## rank %i/%i on %s --\t Building program and creating kernels\n", rank, size-1, hostname);
		char opt[50] = "";
		if (SPMVM_OPTIONS & SPMVM_OPTION_AXPY)
			sprintf(opt+strlen(opt),"-DAXPY ");

		sprintf(opt+strlen(opt),"-DT=");
		sprintf(opt+strlen(opt),"%d",matrixFormats->T[i]);

		CL_safecall(clBuildProgram(program[i],1,&deviceIDs[takedevice],opt,NULL,NULL));

		IF_DEBUG(1) {
			CL_safecall(clGetProgramBuildInfo(program[i],deviceIDs[takedevice],CL_PROGRAM_BUILD_LOG,0,NULL,&log_size));
			build_log = (char *)allocateMemory(log_size+1,"build log");
			CL_safecall(clGetProgramBuildInfo(program[i],deviceIDs[takedevice],CL_PROGRAM_BUILD_LOG,log_size,build_log,NULL));
			printf("Build log: %s",build_log);
		}

		char kernelName[50]="";
		strcat(kernelName, matrixFormats->format[i]==SPM_FORMAT_ELR?"ELR":"pJDS");
		if (matrixFormats->T[i] > 1)
			strcat(kernelName,"T");
		strcat(kernelName,"kernel");
		if (i==2)
			strcat(kernelName,"Add");

		kernel[i] = clCreateKernel(program[i],kernelName,&err);
		CL_checkerror(err);
	}

	kernel[AXPY_KERNEL] = clCreateKernel(program[0],"axpyKernel",&err);
	kernel[DOTPROD_KERNEL] = clCreateKernel(program[0],"dotprodKernel",&err);
	kernel[VECSCAL_KERNEL] = clCreateKernel(program[0],"vecscalKernel",&err);



	free(deviceIDs);
	free(platformIDs);
}


/* *********** CUDA MEMORY **************************** */

cl_mem CL_allocDeviceMemoryMapped( size_t bytesize, void *hostPtr ) {
	cl_mem mem;
	cl_int err;

	mem = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,bytesize,hostPtr,&err);

	CL_checkerror(err);

	return mem;
}
cl_mem CL_allocDeviceMemory( size_t bytesize ) {
	cl_mem mem;
	cl_int err;

	mem = clCreateBuffer(context,CL_MEM_READ_WRITE,bytesize,NULL,&err);

	CL_checkerror(err);

	return mem;
}

void * CL_mapBuffer(cl_mem devmem, size_t bytesize) {
	cl_int err;
	void * ret = clEnqueueMapBuffer(queue,devmem,CL_TRUE,CL_MAP_WRITE,0,bytesize,0,NULL,NULL,&err);
	CL_checkerror(err);
	return ret;


}

void* allocHostMemory( size_t sz) {
	return allocateMemory(sz,"allocHostMemory");
}

inline void CL_copyDeviceToHost( void* hostmem, cl_mem devmem, size_t bytesize ) {
	CL_safecall(clEnqueueReadBuffer(queue,devmem,CL_TRUE,0,bytesize,hostmem,0,NULL,NULL));
}
inline cl_event CL_copyDeviceToHostNonBlocking( void* hostmem, cl_mem devmem, size_t bytesize ) {
	cl_event event;
	CL_safecall(clEnqueueReadBuffer(queue,devmem,CL_FALSE,0,bytesize,hostmem,0,NULL,&event));
	return event;

}

inline void CL_copyHostToDeviceOffset( cl_mem devmem, void *hostmem, size_t bytesize, size_t offset ) {
	CL_safecall(clEnqueueWriteBuffer(queue,devmem,CL_TRUE,offset,bytesize,hostmem,0,NULL,NULL));
}

inline void CL_copyHostToDevice( cl_mem devmem, void *hostmem, size_t bytesize ) {
	CL_copyHostToDeviceOffset(devmem, hostmem, bytesize, 0);
}



void CL_freeDeviceMemory( cl_mem mem ) {
	CL_safecall(clReleaseMemObject(mem));
}

void freeHostMemory( void *mem ) {
	free(mem);
}

void CL_bindMatrixToKernel(void *mat, int format, int T, int kernelIdx) 
{
	if (mat == NULL)
		return;

	if (format == SPM_FORMAT_ELR) {
		CL_ELR_TYPE *matrix = (CL_ELR_TYPE *)mat;
		globalSize[kernelIdx] = (size_t)matrix->padding*T;

		CL_safecall(clSetKernelArg(kernel[kernelIdx],2,sizeof(int),   &matrix->nRows));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],3,sizeof(int),   &matrix->padding));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],4,sizeof(cl_mem),&matrix->val));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],5,sizeof(cl_mem),&matrix->col));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],6,sizeof(cl_mem),&matrix->rowLen));
		if (T>1)
			CL_safecall(clSetKernelArg(kernel[kernelIdx],7,sizeof(double)*localSize[kernelIdx],NULL));
		globalSize[AXPY_KERNEL] = matrix->padding;
		globalSize[VECSCAL_KERNEL] = matrix->padding;
		globalSize[DOTPROD_KERNEL] = matrix->padding;
	} else {
		CL_PJDS_TYPE *matrix = (CL_PJDS_TYPE *)mat;
		globalSize[kernelIdx] = (size_t)matrix->padding*T;

		CL_safecall(clSetKernelArg(kernel[kernelIdx],2,sizeof(int),   &matrix->nRows));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],3,sizeof(cl_mem),&matrix->val));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],4,sizeof(cl_mem),&matrix->col));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],5,sizeof(cl_mem),&matrix->rowLen));
		CL_safecall(clSetKernelArg(kernel[kernelIdx],6,sizeof(cl_mem),&matrix->colStart));
		if (T>1)
			CL_safecall(clSetKernelArg(kernel[kernelIdx],7,sizeof(double)*localSize[kernelIdx],NULL));
		globalSize[AXPY_KERNEL] = matrix->padding;
		globalSize[VECSCAL_KERNEL] = matrix->padding;
		globalSize[DOTPROD_KERNEL] = matrix->padding;
	}
}




void CL_SpMVM(cl_mem rhsVec, cl_mem resVec, int type) 
{
	CL_safecall(clSetKernelArg(kernel[type],0,sizeof(cl_mem),&resVec));
	CL_safecall(clSetKernelArg(kernel[type],1,sizeof(cl_mem),&rhsVec));

	CL_safecall(clEnqueueNDRangeKernel(queue,kernel[type],1,NULL,&globalSize[type],&localSize[type],0,NULL,NULL));
}

void CL_vecscal(cl_mem a, double s, int nRows) 
{
	CL_safecall(clSetKernelArg(kernel[VECSCAL_KERNEL],0,sizeof(cl_mem),&a));
	CL_safecall(clSetKernelArg(kernel[VECSCAL_KERNEL],1,sizeof(double),&s));
	CL_safecall(clSetKernelArg(kernel[VECSCAL_KERNEL],2,sizeof(int),&nRows));

	CL_safecall(clEnqueueNDRangeKernel(queue,kernel[VECSCAL_KERNEL],1,NULL,&globalSize[VECSCAL_KERNEL],&localSize[VECSCAL_KERNEL],0,NULL,NULL));
}

void CL_axpy(cl_mem a, cl_mem b, double s, int nRows) 
{
	CL_safecall(clSetKernelArg(kernel[AXPY_KERNEL],0,sizeof(cl_mem),&a));
	CL_safecall(clSetKernelArg(kernel[AXPY_KERNEL],1,sizeof(cl_mem),&b));
	CL_safecall(clSetKernelArg(kernel[AXPY_KERNEL],2,sizeof(double),&s));
	CL_safecall(clSetKernelArg(kernel[AXPY_KERNEL],3,sizeof(int),&nRows));

	CL_safecall(clEnqueueNDRangeKernel(queue,kernel[AXPY_KERNEL],1,NULL,&globalSize[AXPY_KERNEL],&localSize[AXPY_KERNEL],0,NULL,NULL));
}

void CL_dotprod(cl_mem a, cl_mem b, double *out, int nRows) 
{
	int resVecSize = globalSize[DOTPROD_KERNEL]/localSize[DOTPROD_KERNEL]; 
	int i;
	*out = 0.0;

	VECTOR_TYPE *tmp = newVector(resVecSize*sizeof(double));

	CL_safecall(clSetKernelArg(kernel[DOTPROD_KERNEL],0,sizeof(cl_mem),&a));
	CL_safecall(clSetKernelArg(kernel[DOTPROD_KERNEL],1,sizeof(cl_mem),&b));
	CL_safecall(clSetKernelArg(kernel[DOTPROD_KERNEL],2,sizeof(cl_mem),&tmp->CL_val_gpu));
	CL_safecall(clSetKernelArg(kernel[DOTPROD_KERNEL],3,sizeof(int),&nRows));
	CL_safecall(clSetKernelArg(kernel[DOTPROD_KERNEL],4,sizeof(double)*localSize[DOTPROD_KERNEL],NULL));

	CL_safecall(clEnqueueNDRangeKernel(queue,kernel[DOTPROD_KERNEL],1,NULL,&globalSize[DOTPROD_KERNEL],&localSize[DOTPROD_KERNEL],0,NULL,NULL));

	CL_copyDeviceToHost(tmp->val,tmp->CL_val_gpu,resVecSize*sizeof(double));

	for(i = 0; i < resVecSize; ++i) {
		*out += tmp->val[i];
	}
	freeVector(tmp);
}


void CL_finish() 
{

	int i;
	
	for (i=0; i<6; i++) 
		CL_safecall(clReleaseKernel(kernel[i]));

	CL_safecall(clReleaseCommandQueue(queue));
	CL_safecall(clReleaseContext(context));


}

void CL_uploadCRS(LCRP_TYPE *lcrp, MATRIX_FORMATS *matrixFormats) {

	int node_rank, node_size;
	int ierr;
	int me;
	int i;
	char hostname[MAXHOSTNAMELEN];
	int me_node;



	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );
	gethostname(hostname,MAXHOSTNAMELEN);

	ierr = MPI_Comm_size( single_node_comm, &node_size);
	ierr = MPI_Comm_rank( single_node_comm, &node_rank);
	CL_init( node_rank, node_size, hostname, matrixFormats);
	CL_setup_communication(lcrp,matrixFormats);
	
	if( JOBMASK & 503 ) { // only if jobtype requires combined computation
		CL_bindMatrixToKernel(lcrp->fullMatrix,lcrp->fullFormat,matrixFormats->T[SPM_KERNEL_FULL],SPM_KERNEL_FULL);
	}
	
	if( JOBMASK & 261640 ) { // only if jobtype requires split computation
		CL_bindMatrixToKernel(lcrp->localMatrix,lcrp->localFormat,matrixFormats->T[SPM_KERNEL_LOCAL],SPM_KERNEL_LOCAL);
		CL_bindMatrixToKernel(lcrp->remoteMatrix,lcrp->remoteFormat,matrixFormats->T[SPM_KERNEL_REMOTE],SPM_KERNEL_REMOTE);
	}


}

void CL_setup_communication(LCRP_TYPE* lcrp, MATRIX_FORMATS *matrixFormats){

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


	int ierr, me;

	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);
	IF_DEBUG(1) printf("PE%i: creating matrices:\n", me);


	if( JOBMASK & 503 ) { // only if jobtype requires combined computation


		switch (matrixFormats->format[0]) {
			case SPM_FORMAT_PJDS:
				{
					IF_DEBUG(1) printf("PE%i: FULL pjds:\n", me);

					pjds = CRStoPJDST( lcrp->val, lcrp->col, lcrp->lrow_ptr, lcrp->lnRows[me],matrixFormats->T[0] );
					lcrp->fullRowPerm = (int *)allocateMemory(sizeof(int)*lcrp->lnRows[me],"rowPerm");
					lcrp->fullInvRowPerm = (int *)allocateMemory(sizeof(int)*lcrp->lnRows[me],"invRowPerm");
					memcpy(lcrp->fullRowPerm, pjds->rowPerm, lcrp->lnRows[me]*sizeof(int));
					memcpy(lcrp->fullInvRowPerm, pjds->invRowPerm, lcrp->lnRows[me]*sizeof(int));

					cpjds = CL_initPJDS( pjds );
					CL_uploadPJDS(cpjds, pjds);
					lcrp->fullMatrix = cpjds;
					lcrp->fullFormat = SPM_FORMAT_PJDS;

					/*IF_DEBUG(1) {
					  resetPJDS( pjds );
					  cudaCopyPJDSBackToHost( pjds, cpjds );
					  checkCRSToPJDSsanity( lcrp->val, lcrp->col, lcrp->lrow_ptr, lcrp->lnRows[me], pjds, lcrp->invRowPerm );
					  }*/

					freePJDS( pjds );
					break;
				}
			case SPM_FORMAT_ELR:
				{

					IF_DEBUG(1) printf("PE%i: FULL elr-%d:\n", me,matrixFormats->T[0]);

					elr = CRStoELRT( lcrp->val, lcrp->col, lcrp->lrow_ptr, lcrp->lnRows[me],matrixFormats->T[0] );
					celr = CL_initELR( elr );
					CL_uploadELR(celr, elr);
					lcrp->fullMatrix = celr;
					lcrp->fullFormat = SPM_FORMAT_ELR;
/*					IF_DEBUG(1) {
						resetELR( elr );
						CL_CopyELRBackToHost( elr, celr );
						checkCRSToELRsanity( lcrp->val, lcrp->col, lcrp->lrow_ptr, lcrp->lnRows[me], elr );
					}*/

					freeELR( elr );
					break;
				}

		}

	}

	if( JOBMASK & 261640 ) { // only if jobtype requires split computation

		if (matrixFormats->format[1] == SPM_FORMAT_PJDS) {
					IF_DEBUG(1) printf("PE%i: LOCAL pjds:\n", me);

					lpjds = CRStoPJDST( lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l, lcrp->lnRows[me],matrixFormats->T[1] );

					lcrp->splitRowPerm = (int *)allocateMemory(sizeof(int)*lcrp->lnRows[me],"rowPerm");
					lcrp->splitInvRowPerm = (int *)allocateMemory(sizeof(int)*lcrp->lnRows[me],"invRowPerm");
					memcpy(lcrp->splitRowPerm, lpjds->rowPerm, lcrp->lnRows[me]*sizeof(int));
					memcpy(lcrp->splitInvRowPerm, lpjds->invRowPerm, lcrp->lnRows[me]*sizeof(int));

					lcpjds = CL_initPJDS( lpjds );
					CL_uploadPJDS(lcpjds, lpjds);
					lcrp->localMatrix = lcpjds;
					lcrp->localFormat = SPM_FORMAT_PJDS;

					/*		IF_DEBUG(1) {
							resetPJDS( lpjds );
							cudaCopyPJDSBackToHost( lpjds, lcpjds );
							checkCRSToPJDSsanity( lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l, lcrp->lnRows[me], lpjds, lcrp->invRowPerm->val );
							}*/
					freePJDS( lpjds );
		}
		if (matrixFormats->format[2] == SPM_FORMAT_PJDS) {
					IF_DEBUG(1) printf("PE%i: REMOTE pjds:\n", me);

					rpjds = CRStoPJDST( lcrp->rval, lcrp->rcol, lcrp->lrow_ptr_r, lcrp->lnRows[me],matrixFormats->T[2] );

					lcrp->splitRowPerm = (int *)allocateMemory(sizeof(int)*lcrp->lnRows[me],"rowPerm");
					lcrp->splitInvRowPerm = (int *)allocateMemory(sizeof(int)*lcrp->lnRows[me],"invRowPerm");
					memcpy(lcrp->splitRowPerm, rpjds->rowPerm, lcrp->lnRows[me]*sizeof(int));
					memcpy(lcrp->splitInvRowPerm, rpjds->invRowPerm, lcrp->lnRows[me]*sizeof(int));


					rcpjds = CL_initPJDS( rpjds );
					CL_uploadPJDS(rcpjds, rpjds);
					lcrp->remoteMatrix = rcpjds;
					lcrp->remoteFormat = SPM_FORMAT_PJDS;

					/*		IF_DEBUG(1) {
							resetPJDS( lpjds );
							cudaCopyPJDSBackToHost( lpjds, lcpjds );
							checkCRSToPJDSsanity( lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l, lcrp->lnRows[me], lpjds, lcrp->invRowPerm->val );
							}*/
					freePJDS( rpjds );


		}
		if (matrixFormats->format[1] == SPM_FORMAT_ELR) {
					IF_DEBUG(1) printf("PE%i: LOCAL elr:\n", me);

					if (matrixFormats->format[2] == SPM_FORMAT_PJDS)
						lelr = CRStoELRTP( lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l, lcrp->lnRows[me],lcrp->splitRowPerm,lcrp->splitInvRowPerm,matrixFormats->T[1] );
					else
						lelr = CRStoELRT( lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l, lcrp->lnRows[me],matrixFormats->T[1] );

					lcelr = CL_initELR( lelr );
					CL_uploadELR(lcelr, lelr);
					lcrp->localMatrix = lcelr;
					lcrp->localFormat = SPM_FORMAT_ELR;

/*					IF_DEBUG(1) {
						resetELR( lelr );
						CL_CopyELRBackToHost( lelr, lcelr );
						checkCRSToELRsanity( lcrp->lval, lcrp->lcol, lcrp->lrow_ptr_l, lcrp->lnRows[me], lelr );
					}*/
					freeELR( lelr );
		}
		if (matrixFormats->format[2] == SPM_FORMAT_ELR) {
					IF_DEBUG(1) printf("PE%i: REMOTE elr:\n", me);

					if (matrixFormats->format[1] == SPM_FORMAT_PJDS)
						relr = CRStoELRTP( lcrp->rval, lcrp->rcol, lcrp->lrow_ptr_r, lcrp->lnRows[me],lcrp->splitRowPerm,lcrp->splitInvRowPerm,matrixFormats->T[2] );
					else
						relr = CRStoELRT( lcrp->rval, lcrp->rcol, lcrp->lrow_ptr_r, lcrp->lnRows[me],matrixFormats->T[2] );


					rcelr = CL_initELR( relr );

					CL_uploadELR(rcelr, relr);
					lcrp->remoteMatrix = rcelr;
					lcrp->remoteFormat = SPM_FORMAT_ELR;

					/*IF_DEBUG(1) {
					  resetELR( relr );
					  cudaCopyELRBackToHost( relr, rcelr );
					  checkCRSToELRsanity( lcrp->rval, lcrp->rcol, lcrp->lrow_ptr_r, lcrp->lnRows[me], relr);
					  }*/
					freeELR( relr ); 
		}
	}
}
