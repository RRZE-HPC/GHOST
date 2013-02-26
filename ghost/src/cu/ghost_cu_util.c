#include "ghost_util.h"
#include "ghost_mat.h"
#include <string.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <sys/param.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CU_MAX_DEVICE_NAME_LEN 500


static int device;

void CU_init()
{
	int nDevs;
	CU_safecall(cudaGetDeviceCount(&nDevs));

	if (nDevs < ghost_getNumberOfRanksOnNode()) {
		ABORT("There are more MPI ranks (%d) on the node than CUDA devices available (%d)!",ghost_getNumberOfRanksOnNode(),nDevs);
	}

	device = ghost_getLocalRank();

	DEBUG_LOG(1,"Selecting CUDA device %d",device);
	CU_safecall(cudaSetDevice(device));
}

void * CU_allocDeviceMemory( size_t bytesize )
{
	if (bytesize == 0)
		return NULL;

	void *ret;
	CU_safecall(cudaMalloc(&ret,bytesize));

	return ret;
}

void CU_copyDeviceToHost(void * hostmem, void * devmem, size_t bytesize) 
{
	if (bytesize > 0)
		CU_safecall(cudaMemcpy(hostmem,devmem,bytesize,cudaMemcpyDeviceToHost));
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
	if (bytesize > 0)
		CU_safecall(cudaMemcpy(devmem,hostmem,bytesize,cudaMemcpyHostToDevice));
}

void CU_freeDeviceMemory(void * mem)
{
	CU_safecall(cudaFree(mem));
}

void CU_barrier()
{
	CU_safecall(cudaDeviceSynchronize());
}

void CU_finish() 
{

}


void CU_uploadVector( ghost_vec_t *vec )
{
	CU_copyHostToDevice(vec->CU_val,vec->val,vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
}

void CU_downloadVector( ghost_vec_t *vec )
{
	CU_copyDeviceToHost(vec->val,vec->CU_val,vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
}

static int stringcmp(const void *x, const void *y)
{
	return (strcmp((char *)x, (char *)y));
}

ghost_acc_info_t *CU_getDeviceInfo() 
{
	ghost_acc_info_t *devInfo = allocateMemory(sizeof(ghost_acc_info_t),"devInfo");
	devInfo->nDistinctDevices = 1;

	int me,size,i;
	char name[CU_MAX_DEVICE_NAME_LEN];
	char *names = NULL;

	me = ghost_getRank();
	size = ghost_getNumberOfProcesses();

	struct cudaDeviceProp devProp;

	CU_safecall(cudaGetDeviceProperties(&devProp,device));

	strncpy(name,devProp.name,CU_MAX_DEVICE_NAME_LEN);

	int *displs;
	int *recvcounts;

	if (me==0) {
		names = (char *)allocateMemory(size*CU_MAX_DEVICE_NAME_LEN*sizeof(char),
				"names");
		recvcounts = (int *)allocateMemory(sizeof(int)*ghost_getNumberOfProcesses(),"displs");
		displs = (int *)allocateMemory(sizeof(int)*ghost_getNumberOfProcesses(),"displs");
		
		for (i=0; i<ghost_getNumberOfProcesses(); i++) {
			recvcounts[i] = CU_MAX_DEVICE_NAME_LEN;
			displs[i] = i*CU_MAX_DEVICE_NAME_LEN;

		}
	}


#ifdef MPI
	MPI_safecall(MPI_Gatherv(name,CU_MAX_DEVICE_NAME_LEN,MPI_CHAR,names,
				recvcounts,displs,MPI_CHAR,0,MPI_COMM_WORLD));
#else
	strncpy(names,name,CU_MAX_DEVICE_NAME_LEN);
#endif

	if (me==0) {
		qsort(names,size,CU_MAX_DEVICE_NAME_LEN*sizeof(char),stringcmp);
		for (i=1; i<size; i++) {
			if (strcmp(names+(i-1)*CU_MAX_DEVICE_NAME_LEN,
						names+i*CU_MAX_DEVICE_NAME_LEN)) {
				devInfo->nDistinctDevices++;
			}
		}
	}

#ifdef MPI
	MPI_safecall(MPI_Bcast(&devInfo->nDistinctDevices,1,MPI_INT,0,MPI_COMM_WORLD));
#endif

	devInfo->nDevices = allocateMemory(sizeof(int)*devInfo->nDistinctDevices,"nDevices");
	devInfo->names = allocateMemory(sizeof(char *)*devInfo->nDistinctDevices,"device names");
	for (i=0; i<devInfo->nDistinctDevices; i++) {
		devInfo->names[i] = allocateMemory(sizeof(char)*CU_MAX_DEVICE_NAME_LEN,"device names");
		devInfo->nDevices[i] = 1;
	}

	if (me==0) {
		strncpy(devInfo->names[0],names,CU_MAX_DEVICE_NAME_LEN);

		int distIdx = 1;
		for (i=1; i<size; i++) {
			if (strcmp(names+(i-1)*CU_MAX_DEVICE_NAME_LEN,
						names+i*CU_MAX_DEVICE_NAME_LEN)) {
				strncpy(devInfo->names[distIdx],names+i*CU_MAX_DEVICE_NAME_LEN,CU_MAX_DEVICE_NAME_LEN);
				distIdx++;
			} else {
				devInfo->nDevices[distIdx-1]++;
			}
		}
		free(names);
	}

#ifdef MPI
	MPI_safecall(MPI_Bcast(devInfo->nDevices,devInfo->nDistinctDevices,MPI_INT,0,MPI_COMM_WORLD));

	for (i=0; i<devInfo->nDistinctDevices; i++)
		MPI_safecall(MPI_Bcast(devInfo->names[i],CU_MAX_DEVICE_NAME_LEN,MPI_CHAR,0,MPI_COMM_WORLD));
#endif


	return devInfo;
}

const char * CU_getVersion()
{
	int rtVersion, drVersion;
	CU_safecall(cudaRuntimeGetVersion(&rtVersion));
	CU_safecall(cudaDriverGetVersion(&drVersion));
	char *version = (char *)malloc(1024); // TODO as parameter, else: leak
	snprintf(version,1024,"Runtime: %d, Driver: %d",rtVersion,drVersion);
	return version;
}
