NVCC = nvcc

NVCCFLAGS = -Xcompiler -fpic -arch=sm_20  #-gencode=arch=compute_20,code=sm_20# -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35 
