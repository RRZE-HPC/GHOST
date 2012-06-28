
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#if defined(cl_intel_printf)
#pragma OPENCL EXTENSION cl_intel_printf : enable
#elif defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif


kernel void ELRkernel (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen) {

	int row = get_global_id(0);
	double svalue = 0.0, value;
	int i, idcol;
	if (row < nRows) {


		for( i = 0; i < rowLen[row]; ++i) {
			value = val[i*pad+row];
			idcol = col[i*pad+row];
			svalue += value * rhsVec[idcol];


		}
		resVec[row] += svalue;

	}
}

kernel void ELRkernelAdd (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen) {

	int row = get_global_id(0);
	double svalue = 0.0, value;
	int i, idcol;

	if (row < nRows) {
		for( i = 0; i < rowLen[row]; ++i) {
			value = val[i*pad+row];
			idcol = col[i*pad+row];
			svalue += value * rhsVec[idcol];
		}

		resVec[row] += svalue;
	}
}


kernel void pJDSkernel (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart) {

	int row = get_global_id(0);
	double svalue = 0.0, value;
	int i, idcol;

	if (row < nRows) {
		for( i = 0; i < rowLen[row]; ++i) {
			value = val[colStart[i]+row];
			idcol = col[colStart[i]+row];
			svalue += value * rhsVec[idcol];
		}
		resVec[row] = svalue;
	}

}

kernel void pJDSkernelAdd (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart) {

	int row = get_global_id(0);
	double svalue = 0.0, value;
	int i, idcol;

	if (row < nRows) {
		for( i = 0; i < rowLen[row]; ++i) {
			value = val[colStart[i]+row];
			idcol = col[colStart[i]+row];
			svalue += value * rhsVec[idcol];
		}
		resVec[row] += svalue;
	}
}

kernel void pJDSTkernel (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart, local double *shared) {

	unsigned int row  = get_global_id(0)/T;
	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%T;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*T+idb];
			indcol = col[colStart[k]+row*T+idb];
			svalue += value * rhsVec[indcol];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

#if T>2
#if T>4
#if T>8
#if T>16	
		if (idb<16)
			shared[get_local_id(0)]+=shared[get_local_id(0)+16];
#endif
		if (idb<8)
			shared[get_local_id(0)]+=shared[get_local_id(0)+8];
#endif
		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
#endif
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];
#endif

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void pJDSTkernelAdd (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart, local double *shared) {

	unsigned int row  = get_global_id(0)/T;
	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%T;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*T+idb];
			indcol = col[colStart[k]+row*T+idb];
			svalue += value * rhsVec[indcol];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

#if T>2
#if T>4
#if T>8
#if T>16	
		if (idb<16)
			shared[get_local_id(0)]+=shared[get_local_id(0)+16];
#endif
		if (idb<8)
			shared[get_local_id(0)]+=shared[get_local_id(0)+8];
#endif
		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
#endif
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];
#endif

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELRTkernel (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen, local double *shared) {
	unsigned int row  = get_global_id(0)/T;

	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%T;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*T + T*row + idb]; 
			indcol = col[k*pad*T + T*row + idb]; 
			svalue += value * rhsVec[indcol];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

#if T>2
#if T>4
#if T>8
#if T>16	
		if (idb<16)
			shared[get_local_id(0)]+=shared[get_local_id(0)+16];
#endif
		if (idb<8)
			shared[get_local_id(0)]+=shared[get_local_id(0)+8];
#endif
		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
#endif
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];
#endif

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void ELRTkernelAdd ( global double *resVec, global double *rhsVec,int nRows, int pad, global double *val, global int *col, global int *rowLen, local double *shared) {
	unsigned int row  = get_global_id(0)/T;

	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%T;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*T + T*row + idb]; 
			indcol = col[k*pad*T + T*row + idb]; 
			svalue += value * rhsVec[indcol];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

#if T>2
#if T>4
#if T>8
#if T>16	
		if (idb<16)
			shared[get_local_id(0)]+=shared[get_local_id(0)+16];
#endif
		if (idb<8)
			shared[get_local_id(0)]+=shared[get_local_id(0)+8];
#endif
		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
#endif
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];
#endif

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void axpyKernel(global double *a, global double *b, double s, int nRows)
{
	int i = get_global_id(0); 
	if (i<nRows)
		a[i] = a[i] + s*b[i]; 
}

kernel void vecscalKernel(global double *a, double scal, int nRows)
{
	int i = get_global_id(0);
	if (i<nRows)	
		a[i] = scal * a[i]; 
} 

kernel void dotprodKernel(global double *a, global double *b, global double *out, unsigned int nRows, local volatile double *shared) 
{

	unsigned int tid = get_local_id(0);
	unsigned int i = get_global_id(0);
	shared[tid] = (i < nRows) ? a[i]*b[i] : 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = 1; s < get_local_size(0); s *= 2) {
		if ((tid % (2*s)) == 0) {

			shared[tid] += shared[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (tid == 0)
		out[get_group_id(0)] = shared[0];
}
