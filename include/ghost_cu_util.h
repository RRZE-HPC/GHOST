#ifndef __GHOST_CU_UTIL_H__
#define __GHOST_CU_UTIL_H__

void ghost_CUDA_init(int dev);
void * CU_allocDeviceMemory( size_t bytesize );
void CU_copyDeviceToHost(void * hostmem, void * devmem, size_t bytesize);
void CU_copyHostToDeviceOffset(void * devmem, void *hostmem, size_t bytesize, size_t offset);
void CU_copyHostToDevice(void * devmem, void *hostmem, size_t bytesize);
void CU_freeDeviceMemory(void * mem);
void CU_barrier();
void CU_finish();
void CU_uploadVector( ghost_vec_t *vec );
void CU_downloadVector( ghost_vec_t *vec );
const char * CU_getVersion();
ghost_acc_info_t *CU_getDeviceInfo();

#endif
