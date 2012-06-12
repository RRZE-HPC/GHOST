#include "oclfun.h"

static cl_command_queue queue;
static cl_context context;
static cl_kernel kernels[NUM_KERNELS];

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
	fprintf(stderr,"OpenCL error (via pfn_notify): %s\n",errinfo);
}

void CL_selectDevice( int rank, int size, const char* hostname ) {
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_platform_id *platformIDs;
	cl_device_id *deviceIDs;
	cl_int err;
	cl_program program;
	int platform, device, takedevice;
	char devicename[1024];



	cl_uint dummy;

	CL_safecall(clGetPlatformIDs(0, NULL, &numPlatforms));
	platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id)*numPlatforms); // segfault with aligned mem


	CL_safecall(clGetPlatformIDs(numPlatforms, platformIDs, &dummy));



	for (platform=0; platform<numPlatforms; platform++) {
		CL_safecall(clGetDeviceIDs(platformIDs[platform],CL_DEVTYPE, 0, NULL, &numDevices));

		if (numDevices > 0) { // correct platform has been found
			break;
		}

	}
	deviceIDs = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevices);
	CL_safecall(clGetDeviceIDs(platformIDs[platform],CL_DEVTYPE, numDevices, deviceIDs, &numDevices));

	if ( 0 == rank ) {
		printf("## rank %i/%i on %s --\t Platform: %d, No. devices of desired type: %d\n", 
				rank, size-1, hostname, platform, numDevices);

		for( device = 0; device < numDevices; ++device) {
			CL_safecall(clGetDeviceInfo(deviceIDs[device],CL_DEVICE_NAME,sizeof(devicename),devicename,NULL));
			printf("## rank %i/%i on %s --\t Device %d: %s\n", 
					rank, size-1, hostname, device, devicename);
		}




	}

	takedevice = 1;//rank%numDevices;
	CL_safecall(clGetDeviceInfo(deviceIDs[takedevice],CL_DEVICE_NAME,sizeof(devicename),devicename,NULL));
	printf("## rank %i/%i on %s --\t Selecting device %d: %s\n", rank, size-1, hostname, takedevice, devicename);

	printf("## rank %i/%i on %s --\t Creating context \n", rank, size-1, hostname);
	cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[platform],0};
	context = clCreateContext(cprops,1,&deviceIDs[takedevice],pfn_notify,NULL,&err);
	CL_checkerror(err);

	printf("## rank %i/%i on %s --\t Creating command queue\n", rank, size-1, hostname);
	queue = clCreateCommandQueue(context,deviceIDs[takedevice],CL_QUEUE_PROFILING_ENABLE,&err);
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
	source_str = (char*)malloc(10000);
	source_size = fread( source_str, 1, 10000, fp);
	fclose( fp );

	printf("## rank %i/%i on %s --\t Creating program\n", rank, size-1, hostname);
	program = clCreateProgramWithSource(context,1,(const char **)&source_str,&source_size,&err);
	CL_checkerror(err);

	printf("## rank %i/%i on %s --\t Building program\n", rank, size-1, hostname);
	CL_safecall(clBuildProgram(program,1,&deviceIDs[takedevice],NULL,NULL,NULL));
	CL_safecall(clGetProgramBuildInfo(program,deviceIDs[takedevice],CL_PROGRAM_BUILD_LOG,0,NULL,&log_size));
	build_log = (char *)malloc(log_size+1);
	CL_safecall(clGetProgramBuildInfo(program,deviceIDs[takedevice],CL_PROGRAM_BUILD_LOG,log_size,build_log,NULL));
	printf("Build log: %s",build_log);


	printf("## rank %i/%i on %s --\t Creating kernels\n", rank, size-1, hostname);
	kernels[KERNEL_ELR] = clCreateKernel(program,"ELRkernel",&err);
	CL_checkerror(err);
	kernels[KERNEL_ELR_ADD] = clCreateKernel(program,"ELRkernelAdd",&err);
	CL_checkerror(err);
	kernels[KERNEL_PJDS] = clCreateKernel(program,"pJDSkernel",&err);
	CL_checkerror(err);
	kernels[KERNEL_PJDS_ADD] = clCreateKernel(program,"pJDSkernelAdd",&err);
	CL_checkerror(err);

	free(deviceIDs);
	free(platformIDs);
}


/* *********** CUDA MEMORY **************************** */

cl_mem CL_allocDeviceMemory( size_t bytesize ) {
	cl_mem mem;
	cl_int err;

	mem = clCreateBuffer(context,CL_MEM_READ_WRITE,bytesize,NULL,&err);
	CL_checkerror(err);

	return mem;
}

void* allocHostMemory( size_t sz) {
	void *mem;
	mem = malloc(sz);
	if (!mem) {
		fprintf(stderr,"allocHostMemory failed!\n");
		abort();
	}

	return mem;


}

void CL_copyDeviceToHost( void* hostmem, cl_mem devmem, size_t bytesize ) {
	CL_safecall(clEnqueueReadBuffer(queue,devmem,CL_TRUE,0,bytesize,hostmem,0,NULL,NULL));
}

void CL_copyHostToDeviceOffset( cl_mem devmem, void *hostmem, size_t bytesize, size_t offset ) {
	CL_safecall(clEnqueueWriteBuffer(queue,devmem,CL_TRUE,offset,bytesize,hostmem,0,NULL,NULL));
}

void CL_copyHostToDevice( cl_mem devmem, void *hostmem, size_t bytesize ) {
	CL_copyHostToDeviceOffset(devmem, hostmem, bytesize, 0);
}



void CL_freeDeviceMemory( cl_mem mem ) {
	CL_safecall(clReleaseMemObject(mem));
}

void freeHostMemory( void *mem ) {
	free(mem);
}


void oclKernel(void *mat,  cl_mem rhsVec, cl_mem resVec, bool add, bool elr) {
	cl_kernel kernel;
	size_t global;

	if (elr) {
		if (add) {
			kernel = kernels[KERNEL_ELR_ADD];
		} else {
			kernel = kernels[KERNEL_ELR];
		}

		CL_ELR_TYPE *matrix = (CL_ELR_TYPE *)mat;
		global = (size_t)matrix->padding;

		CL_safecall(clSetKernelArg(kernel,0,sizeof(int),   &matrix->nRows));
		CL_safecall(clSetKernelArg(kernel,1,sizeof(int),   &matrix->padding));
		CL_safecall(clSetKernelArg(kernel,2,sizeof(cl_mem),&resVec));
		CL_safecall(clSetKernelArg(kernel,3,sizeof(cl_mem),&rhsVec));
		CL_safecall(clSetKernelArg(kernel,4,sizeof(cl_mem),&matrix->val));
		CL_safecall(clSetKernelArg(kernel,5,sizeof(cl_mem),&matrix->col));
		CL_safecall(clSetKernelArg(kernel,6,sizeof(cl_mem),&matrix->rowLen));

		CL_safecall(clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,NULL,0,NULL,NULL));
	} else {
		if (add) {
			kernel = kernels[KERNEL_PJDS_ADD];
		} else {
			kernel = kernels[KERNEL_PJDS];
		}

		
		CL_PJDS_TYPE *matrix = (CL_PJDS_TYPE *)mat;
		global = (size_t)matrix->padding;
		
		CL_safecall(clSetKernelArg(kernel,0,sizeof(int),   &matrix->nRows));
		CL_safecall(clSetKernelArg(kernel,1,sizeof(cl_mem),&resVec));
		CL_safecall(clSetKernelArg(kernel,2,sizeof(cl_mem),&rhsVec));
		CL_safecall(clSetKernelArg(kernel,3,sizeof(cl_mem),&matrix->val));
		CL_safecall(clSetKernelArg(kernel,4,sizeof(cl_mem),&matrix->col));
		CL_safecall(clSetKernelArg(kernel,5,sizeof(cl_mem),&matrix->rowLen));
		CL_safecall(clSetKernelArg(kernel,6,sizeof(cl_mem),&matrix->colStart));

		CL_safecall(clFinish(queue));
		CL_safecall(clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,NULL,0,NULL,NULL));
	}


}

void CL_finish() {

	int i;

/*	for (i=0; i<NUM_KERNELS; i++) 
		CL_safecall(clReleaseKernel(kernels[i]));
	CL_safecall(clReleaseCommandQueue(queue));
	CL_safecall(clReleaseContext(context));
*/

}

