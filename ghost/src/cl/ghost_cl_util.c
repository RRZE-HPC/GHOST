#include "ghost_util.h"
#include "ghost_mat.h"
//#include "mpihelper.h"
#include <string.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <sys/param.h>
#include <CL/cl_ext.h>

#define CL_MAX_DEVICE_NAME_LEN 500

static cl_command_queue queue;
static cl_context context;
static cl_platform_id platform;

static void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	UNUSED(private_info);
	UNUSED(cb);
	UNUSED(user_data);
	// TODO call makro
    fprintf(stderr, ANSI_COLOR_RED "OpenCL error at %s:%d, %s\n" ANSI_COLOR_RESET,__FILE__,__LINE__,errinfo);
}

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
	free(platformIDs);
}

/* -----------------------------------------------------------------------------
   Create program inside previously created context (global variable) and 
   build it
   -------------------------------------------------------------------------- */
cl_program CL_registerProgram(const char *filename, const char *additionalOptions)
{
	DEBUG_LOG(1,"Registering OpenCL program in %s",filename);
	cl_program program;
	cl_int err;
	char *build_log;
	size_t log_size;
	cl_device_id deviceID;

	CL_safecall(clGetContextInfo(context,CL_CONTEXT_DEVICES,
				sizeof(cl_device_id),&deviceID,NULL));
	char devicename[CL_MAX_DEVICE_NAME_LEN];
	CL_safecall(clGetDeviceInfo(deviceID,CL_DEVICE_NAME,sizeof(devicename),devicename,NULL));

	char path[PATH_MAX];
	snprintf(path,PATH_MAX,"%s/%s",PLUGINPATH,filename);

	char headerPath[PATH_MAX] = HEADERPATH;
	size_t optionsLen = strlen(headerPath)+4+strlen(additionalOptions);
	char *options = (char *)malloc(optionsLen);
	snprintf(options,optionsLen,"-I%s %s",headerPath, additionalOptions);

	FILE *fp = fopen(path, "r");
	if (!fp)
		ABORT("Failed to load OpenCL kernel file %s",path);

	fseek(fp,0L,SEEK_END);
	long filesize = ftell(fp);
	fseek(fp,0L,SEEK_SET);

	char * source_str = (char*)allocateMemory(filesize,"source");
	fread( source_str, 1, filesize, fp);
	fclose( fp );

	program = clCreateProgramWithSource(context,1,(const char **)&source_str,
			NULL,&err);
	CL_checkerror(err);

	DEBUG_LOG(1,"Building program with \"%s\" and creating kernels",options);

	CL_safecall(clBuildProgram(program,1,&deviceID,options,NULL,NULL));

	IF_DEBUG(1) {
		CL_safecall(clGetProgramBuildInfo(program,deviceID,
					CL_PROGRAM_BUILD_LOG,0,NULL,&log_size));
		build_log = (char *)allocateMemory(log_size+1,"build log");
		CL_safecall(clGetProgramBuildInfo(program,deviceID,
					CL_PROGRAM_BUILD_LOG,log_size,build_log,NULL));
		DEBUG_LOG(1,"Build log: %s",build_log);
	}

	return program;
}

cl_mem CL_allocDeviceMemoryMapped( size_t bytesize, void *hostPtr, int flag )
{
	cl_mem mem;
	cl_int err;

	mem = clCreateBuffer(context,flag|CL_MEM_USE_HOST_PTR,bytesize,
			hostPtr,&err);
	if (!(err == CL_INVALID_BUFFER_SIZE && bytesize == 0))
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

/*cl_mem CL_allocDeviceMemoryCached( size_t bytesize, void *hostPtr )
  {
  cl_mem mem = NULL;
  cl_int err;
  cl_image_format image_format;

  image_format.image_channel_order = CL_RG;
  image_format.image_channel_ghost_mdat_type = CL_FLOAT;

  mem = clCreateImage2D(context,CL_MEM_READ_WRITE,&image_format,bytesize/sizeof(ghost_mdat_t),1,0,hostPtr,&err);

  printf("image width: %lu\n",bytesize/sizeof(ghost_mdat_t));	

  CL_checkerror(err);

  return mem;
  }*/

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
	const size_t region[3] = {bytesize/sizeof(ghost_mdat_t),0,0};
	CL_safecall(clEnqueueReadImage(queue,devmem,CL_TRUE,origin,region,0,0,
				hostmem,0,NULL,NULL));
#else
	DEBUG_LOG(1,"Copying back %lu bytes to host\n",bytesize);
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
	DEBUG_LOG(1,"Copying %lu bytes to device",bytesize);
#ifdef CL_IMAGE
	cl_mem_object_type type;
	CL_safecall(clGetMemObjectInfo(devmem,CL_MEM_TYPE,sizeof(cl_mem_object_type),&type,NULL));
	if (type == CL_MEM_OBJECT_BUFFER) {
		CL_safecall(clEnqueueWriteBuffer(queue,devmem,CL_TRUE,offset,bytesize,
					hostmem,0,NULL,NULL));
	} else {
		const size_t origin[3] = {offset,0,0};
		const size_t region[3] = {bytesize/sizeof(ghost_mdat_t),0,0};
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
		const size_t region[3] = {bytesize/sizeof(ghost_mdat_t),0,0};
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

void CL_enqueueKernel(cl_kernel kernel, cl_uint dim, size_t *gSize, size_t *lSize)
{
	DEBUG_LOG(1,"Enqueueing kernel with global size %lu, local size %lu",gSize==NULL?0:*gSize,lSize==NULL?0:*lSize);
	CL_safecall(clEnqueueNDRangeKernel(queue,kernel,dim,NULL,gSize,lSize,0,NULL,NULL));
}

void CL_barrier()
{
	CL_safecall(clEnqueueBarrier(queue));
	CL_safecall(clFinish(queue));
}

void CL_finish() 
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

/*void CL_uploadCRS(ghost_mat_t *matrix, GHOST_SPM_GPUFORMATS *matrixFormats, int spmvmOptions)
  {

  if (!(matrix->format & GHOST_SPMFORMAT_DIST_CRS)) {
  DEBUG_LOG(0,"Device matrix can only be created from a distributed CRS host matrix.");
  return;
  }

  CL_createMatrix(matrix,matrixFormats,spmvmOptions);

  if (!(spmvmOptions & GHOST_OPTION_NO_COMBINED_SOLVERS)) { // combined computation
  CL_bindMatrixToKernel(gpum->fullMatrix,gpum->fullFormat,
  matrixFormats->T[GHOST_FULL_MAT_IDX],GHOST_FULL_MAT_IDX, spmvmOptions);
  }

  if (!(spmvmOptions & GHOST_OPTION_NO_SPLIT_SOLVERS)) { // split computation
  CL_bindMatrixToKernel(gpum->localMatrix,gpum->localFormat,
  matrixFormats->T[GHOST_LOCAL_MAT_IDX],GHOST_LOCAL_MAT_IDX, spmvmOptions);
  CL_bindMatrixToKernel(gpum->remoteMatrix,gpum->remoteFormat,
  matrixFormats->T[GHOST_REMOTE_MAT_IDX],GHOST_REMOTE_MAT_IDX, spmvmOptions);
  }

  return gpum;

  }*/

/*void CL_createMatrix(ghost_mat_t* matrix, GHOST_SPM_GPUFORMATS *matrixFormats, int spmvmOptions)
  {

  ghost_comm_t *lcrp = (ghost_comm_t *)matrix->matrix;
  GPUghost_mat_t * gpum = (GPUghost_mat_t*) allocateMemory( sizeof( GPUghost_mat_t ), "gpum" );

  int me = ghost_getRank();

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


  if (!(spmvmOptions & GHOST_OPTION_NO_COMBINED_SOLVERS)) { // combined computation
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

if (!(spmvmOptions & GHOST_OPTION_NO_SPLIT_SOLVERS)) { // split computation
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
}*/

void CL_uploadVector( ghost_vec_t *vec )
{
	CL_copyHostToDevice(vec->CL_val_gpu,vec->val,vec->nrows*sizeof(ghost_vdat_t));
}

void CL_downloadVector( ghost_vec_t *vec )
{
	CL_copyDeviceToHost(vec->val,vec->CL_val_gpu,vec->nrows*sizeof(ghost_vdat_t));
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

ghost_cl_devinfo_t *CL_getDeviceInfo() 
{
	ghost_cl_devinfo_t *devInfo = allocateMemory(sizeof(ghost_cl_devinfo_t),"devInfo");
	devInfo->nDistinctDevices = 1;

	int me,size,i;
	cl_device_id deviceID;
	char name[CL_MAX_DEVICE_NAME_LEN];
	char *names = NULL;

	me = ghost_getRank();
	size = ghost_getNumberOfProcesses();

	CL_safecall(clGetContextInfo(context,CL_CONTEXT_DEVICES,
				sizeof(cl_device_id),&deviceID,NULL));
	CL_safecall(clGetDeviceInfo(deviceID,CL_DEVICE_NAME,
				CL_MAX_DEVICE_NAME_LEN*sizeof(char),name,NULL));

	if (me==0) {
		names = (char *)allocateMemory(size*CL_MAX_DEVICE_NAME_LEN*sizeof(char),
				"names");
	}


#ifdef MPI
	MPI_safecall(MPI_Gather(name,CL_MAX_DEVICE_NAME_LEN,MPI_CHAR,names,
				CL_MAX_DEVICE_NAME_LEN,MPI_CHAR,0,MPI_COMM_WORLD));
#else
	strncpy(names,name,CL_MAX_DEVICE_NAME_LEN);
#endif

	if (me==0) {
		qsort(names,size,CL_MAX_DEVICE_NAME_LEN*sizeof(char),stringcmp);
		for (i=1; i<size; i++) {
			if (strcmp(names+(i-1)*CL_MAX_DEVICE_NAME_LEN,
						names+i*CL_MAX_DEVICE_NAME_LEN)) {
				devInfo->nDistinctDevices++;
			}
		}
	}

#ifdef MPI
	MPI_safecall(MPI_Bcast(&devInfo->nDistinctDevices,1,MPI_INT,0,MPI_COMM_WORLD));
#endif

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

#ifdef MPI
	MPI_safecall(MPI_Bcast(devInfo->nDevices,devInfo->nDistinctDevices,MPI_INT,0,MPI_COMM_WORLD));

	for (i=0; i<devInfo->nDistinctDevices; i++)
		MPI_safecall(MPI_Bcast(devInfo->names[i],CL_MAX_DEVICE_NAME_LEN,MPI_CHAR,0,MPI_COMM_WORLD));
#endif


	return devInfo;
}

void destroyCLdeviceInfo(ghost_cl_devinfo_t * di) 
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

// copyright NVIDIA SDK
const char * CL_errorString(cl_int err)
{

	static const char* errorString[] = {
		"CL_SUCCESS",
		"CL_DEVICE_NOT_FOUND",
		"CL_DEVICE_NOT_AVAILABLE",
		"CL_COMPILER_NOT_AVAILABLE",
		"CL_MEM_OBJECT_ALLOCATION_FAILURE",
		"CL_OUT_OF_RESOURCES",
		"CL_OUT_OF_HOST_MEMORY",
		"CL_PROFILING_INFO_NOT_AVAILABLE",
		"CL_MEM_COPY_OVERLAP",
		"CL_IMAGE_FORMAT_MISMATCH",
		"CL_IMAGE_FORMAT_NOT_SUPPORTED",
		"CL_BUILD_PROGRAM_FAILURE",
		"CL_MAP_FAILURE",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"CL_INVALID_VALUE",
		"CL_INVALID_DEVICE_TYPE",
		"CL_INVALID_PLATFORM",
		"CL_INVALID_DEVICE",
		"CL_INVALID_CONTEXT",
		"CL_INVALID_QUEUE_PROPERTIES",
		"CL_INVALID_COMMAND_QUEUE",
		"CL_INVALID_HOST_PTR",
		"CL_INVALID_MEM_OBJECT",
		"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
		"CL_INVALID_IMAGE_SIZE",
		"CL_INVALID_SAMPLER",
		"CL_INVALID_BINARY",
		"CL_INVALID_BUILD_OPTIONS",
		"CL_INVALID_PROGRAM",
		"CL_INVALID_PROGRAM_EXECUTABLE",
		"CL_INVALID_KERNEL_NAME",
		"CL_INVALID_KERNEL_DEFINITION",
		"CL_INVALID_KERNEL",
		"CL_INVALID_ARG_INDEX",
		"CL_INVALID_ARG_VALUE",
		"CL_INVALID_ARG_SIZE",
		"CL_INVALID_KERNEL_ARGS",
		"CL_INVALID_WORK_DIMENSION",
		"CL_INVALID_WORK_GROUP_SIZE",
		"CL_INVALID_WORK_ITEM_SIZE",
		"CL_INVALID_GLOBAL_OFFSET",
		"CL_INVALID_EVENT_WAIT_LIST",
		"CL_INVALID_EVENT",
		"CL_INVALID_OPERATION",
		"CL_INVALID_GL_OBJECT",
		"CL_INVALID_BUFFER_SIZE",
		"CL_INVALID_MIP_LEVEL",
		"CL_INVALID_GLOBAL_WORK_SIZE",
	};

	const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

	const int index = -err;

	return (index >= 0 && index < errorCount) ? errorString[index] : "Unspecified Error";
}

const char * CL_getVersion()
{
	char *version = (char *)malloc(1024); // TODO as parameter, else: leak
	CL_safecall(clGetPlatformInfo(platform,CL_PLATFORM_VERSION,1024,version,NULL));
	return version;
}


