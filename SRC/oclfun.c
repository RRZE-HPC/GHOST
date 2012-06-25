#include "oclfun.h"
#include <string.h>

static cl_command_queue queue;
static cl_context context;
static cl_kernel kernel[3];
static size_t localSize[3] = {256,256,256};
static size_t globalSize[3];

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
	source_str = (char*)allocateMemory(10000,"source");
	source_size = fread( source_str, 1, 10000, fp);
	fclose( fp );



	int i;
	for (i=0; i<3; i++) {
		IF_DEBUG(1) printf("## rank %i/%i on %s --\t Creating program\n", rank, size-1, hostname);
		program[i] = clCreateProgramWithSource(context,1,(const char **)&source_str,&source_size,&err);
		CL_checkerror(err);


		IF_DEBUG(1) printf("## rank %i/%i on %s --\t Building program and creating kernels\n", rank, size-1, hostname);
		char opt[7];
		strcpy(opt,"-DT=");
		sprintf(opt+4,"%d",matrixFormats->T[i]);
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
	return allocateMemory(sz,"allocHostMemory");
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

void CL_bindMatrixToKernel(void *mat, int format, int T, int kernelIdx) {
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
	}
}




void CL_SpMVM(cl_mem rhsVec, cl_mem resVec, int type) {
	CL_safecall(clSetKernelArg(kernel[type],0,sizeof(cl_mem),&resVec));
	CL_safecall(clSetKernelArg(kernel[type],1,sizeof(cl_mem),&rhsVec));

	CL_safecall(clEnqueueNDRangeKernel(queue,kernel[type],1,NULL,&globalSize[type],&localSize[type],0,NULL,NULL));
}

void CL_finish() {

	int i;

	for (i=0; i<3; i++) 
		CL_safecall(clReleaseKernel(kernel[i]));

	CL_safecall(clReleaseCommandQueue(queue));
	CL_safecall(clReleaseContext(context));


}

