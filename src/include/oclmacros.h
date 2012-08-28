#ifndef _OCL_MACROS_H_
#define _OCL_MACROS_H_


#define CL_safecall(call) {\
  cl_int clerr = call ;\
  if( CL_SUCCESS != clerr ){\
    fprintf(stderr, "OpenCL error at %s:%d, %d\n",\
      __FILE__, __LINE__, clerr);\
    fflush(stderr);\
  }\
  }

#define CL_checkerror(err) do{\
  if( CL_SUCCESS != err ){\
    fprintf(stdout, "OpenCL error at %s:%d, %d\n",\
      __FILE__, __LINE__, err);\
    fflush(stdout);\
  }\
  } while(0)

typedef struct{
	int nDistinctDevices;
	int *nDevices;
	char **names;
} CL_DEVICE_INFO;



#endif
