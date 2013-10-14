NVCC = nvcc

NVCCFLAGS = -O3 -Xcompiler -fpic -gencode=arch=compute_20,code=sm_20# -gencode=arch=compute_30,code=compute_30 #-gencode=arch=compute_35,code=compute_35 #-gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35 
