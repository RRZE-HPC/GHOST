#include "ghost_util.h"
#include "ghost_mat.h"
#include <string.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <sys/param.h>
#include <cuda_runtime.h>

#define CL_MAX_DEVICE_NAME_LEN 500

void CU_init()
{
	/*cl_uint numPlatforms;
	cl_platform_id *platformIDs;
	cl_int err;
	unsigned int platformNum, device;
	char devicename[CL_MAX_DEVICE_NAME_LEN];
	int takedevice;
	int rank = ghost_getLocalRank();
	cl_uint numDevices = 0;
	cl_device_id *deviceIDs;
	cl_device_id deviceID;



	CL_safecall(clGetPlatformIDs(0, NULL, &numPlatforms));
	DEBUG_LOG(1,"There are %u OpenCL platforms",numPlatforms);
	platformIDs = (cl_platform_id *)allocateMemory(
			sizeof(cl_platform_id)*numPlatforms,"platformIDs");
	CL_safecall(clGetPlatformIDs(numPlatforms, platformIDs, NULL));

	for (platformNum=0; platformNum<numPlatforms; platformNum++) {
		err = clGetDeviceIDs(platformIDs[platformNum],CL_MY_DEVICE_TYPE, 0, NULL,
				&numDevices);
		if (err != CL_DEVICE_NOT_FOUND) // not an actual error => no print
			CL_checkerror(err);

		if (numDevices > 0) { // correct platform has been found
			platform = platformIDs[platformNum];
			break;
		}
	}
	if (numDevices == 0)
		ABORT("No suitable OpenCL device found.");

	deviceIDs = (cl_device_id *)allocateMemory(sizeof(cl_device_id)*numDevices,
			"deviceIDs");
	CL_safecall(clGetDeviceIDs(platformIDs[platformNum],CL_MY_DEVICE_TYPE, numDevices,
				deviceIDs, &numDevices));

	DEBUG_LOG(1,"OpenCL platform: %u, no. devices of desired type: %u", platformNum, numDevices);

	IF_DEBUG(1) {
		for( device = 0; device < numDevices; ++device) {
			CL_safecall(clGetDeviceInfo(deviceIDs[device],CL_DEVICE_NAME,
						sizeof(devicename),devicename,NULL));
			DEBUG_LOG(1,"Device %u: %s", device, devicename);
		}
	}

	takedevice = rank%numDevices;
	deviceID = deviceIDs[takedevice];

	CL_safecall(clGetDeviceInfo(deviceID,CL_DEVICE_NAME,sizeof(devicename),devicename,NULL));
	DEBUG_LOG(1,"Selecting device %d: %s", takedevice, devicename);

#ifdef GHOST_CL_DEVICE_FISSION
	DEBUG_LOG(1,"Create subdevices...");
	cl_uint numSubDevices;
	cl_device_partition_property_ext props[3];
	props[0] = CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT;
	props[1] = CL_AFFINITY_DOMAIN_L3_CACHE_EXT;
	props[2] = 0;

	CL_safecall(clCreateSubDevicesEXT(deviceID,props,0,NULL,&numSubDevices));
	DEBUG_LOG(1,"There are %u subdevices",numSubDevices);

	cl_device_id *subdeviceIDs = (cl_device_id *)malloc(numSubDevices*sizeof(cl_device_id));
	CL_safecall(clCreateSubDevicesEXT(deviceID,props,numSubDevices,subdeviceIDs,&numSubDevices));

	deviceID = subdeviceIDs[0]
#endif

	cl_uint nCores;
	CL_safecall(clGetDeviceInfo(deviceID,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&nCores,NULL));
	DEBUG_LOG(1,"The (sub-)device has %u cores",nCores);


	DEBUG_LOG(1,"Creating OpenCL context...");
	cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM,(cl_context_properties)platform,0};
	context = clCreateContext(cprops,1,&deviceID,&pfn_notify,NULL,&err);
	CL_checkerror(err);


	DEBUG_LOG(1,"Creating OpenCL command queue...");
	queue = clCreateCommandQueue(context,deviceID,
			CL_QUEUE_PROFILING_ENABLE|CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
			&err);
	CL_checkerror(err);


	free(deviceIDs);
	free(platformIDs);*/
}

void * CU_allocDeviceMemory( size_t bytesize )
{
	if (bytesize == 0)
		return NULL;

	void *ret;
	cudaMalloc(&ret,bytesize);

	return ret;
}

void CU_copyDeviceToHost(void * hostmem, void * devmem, size_t bytesize) 
{
	cudaMemcpy(hostmem,devmem,bytesize,cudaMemcpyDeviceToHost);
}

void CU_copyHostToDeviceOffset(void * devmem, void *hostmem,
		size_t bytesize, size_t offset)
{
	UNUSED(devmem);
	UNUSED(hostmem);
	UNUSED(bytesize);
	UNUSED(offset);
	ABORT("not implemented yet");
}

void CU_copyHostToDevice(void * devmem, void *hostmem, size_t bytesize)
{
	cudaMemcpy(devmem,hostmem,bytesize,cudaMemcpyHostToDevice);
}

void CU_freeDeviceMemory(void * mem)
{
	cudaFree(mem);
}

void CU_barrier()
{
	/*CL_safecall(clEnqueueBarrier(queue));
	CL_safecall(clFinish(queue));*/
}

void CU_finish() 
{

	// TODO
	/*	if (!(spmvmOptions & GHOST_OPTION_NO_COMBINED_SOLVERS)) {
		CL_safecall(clReleaseKernel(kernel[GHOST_FULL_MAT_IDX]));
		}
		if (!(spmvmOptions & GHOST_OPTION_NO_SPLIT_SOLVERS)) {
		CL_safecall(clReleaseKernel(kernel[GHOST_LOCAL_MAT_IDX]));
		CL_safecall(clReleaseKernel(kernel[GHOST_REMOTE_MAT_IDX]));

		}

		CL_safecall(clReleaseCommandQueue(queue));
		CL_safecall(clReleaseContext(context));*/
}


void CU_uploadVector( ghost_vec_t *vec )
{
	CU_copyHostToDevice(vec->CU_val,vec->val,vec->nrows*sizeof(ghost_vdat_t));
}

void CU_downloadVector( ghost_vec_t *vec )
{
	CU_copyDeviceToHost(vec->val,vec->CU_val,vec->nrows*sizeof(ghost_vdat_t));
}

const char * CU_getVersion()
{
	char *version = (char *)malloc(1024); // TODO as parameter, else: leak
//	CL_safecall(clGetPlatformInfo(platform,CL_PLATFORM_VERSION,1024,version,NULL));
	return version;
}


