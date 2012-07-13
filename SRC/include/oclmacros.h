#ifndef _OCL_MACROS_H_
#define _OCL_MACROS_H_


#define CL_safecall(call) do{\
  cl_int err = call ;\
  if( CL_SUCCESS != err ){\
    fprintf(stdout, "OpenCL error at %s:%d, %d\n",\
      __FILE__, __LINE__, err);\
    fflush(stdout);\
  }\
  } while(0)

#define CL_checkerror(err) do{\
  if( CL_SUCCESS != err ){\
    fprintf(stdout, "OpenCL error at %s:%d, %d\n",\
      __FILE__, __LINE__, err);\
    fflush(stdout);\
  }\
  } while(0)



#define CL_DEVTYPE CL_DEVICE_TYPE_GPU


#endif
