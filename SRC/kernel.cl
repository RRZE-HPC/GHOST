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

kernel void pJDS1kernel (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart) {

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

kernel void ELR1kernel (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen) {

	int row = get_global_id(0);
	double svalue = 0.0, value;
	int i, idcol;
	if (row < nRows) {


		for( i = 0; i < rowLen[row]; ++i) {
			value = val[i*pad+row];
			idcol = col[i*pad+row];
			svalue += value * rhsVec[idcol];


		}
		resVec[row] = svalue;
	}
}

kernel void pJDS2kernel (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart, local double *shared) {

	unsigned int row  = get_global_id(0)>>1;
	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%2;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*2+idb];
			indcol = col[colStart[k]+row*2+idb];
			svalue += value * rhsVec[indcol];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELR2kernel (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen, local double *shared) {
	unsigned int row  = get_global_id(0)>>1;

	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%2;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*2 + 2*row + idb]; 
			indcol = col[k*pad*2 + 2*row + idb]; 
			svalue += value * rhsVec[indcol];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void pJDS4kernel (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart, local double *shared) {

	unsigned int row  = get_global_id(0)>>2;
	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%4;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*4+idb];
			indcol = col[colStart[k]+row*4+idb];
			svalue += value * rhsVec[indcol];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELR4kernel (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen, local double *shared) {
	unsigned int row  = get_global_id(0)>>2;

	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%4;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*4 + 4*row + idb]; 
			indcol = col[k*pad*4 + 4*row + idb]; 
			svalue += value * rhsVec[indcol];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void pJDS8kernel (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart, local double *shared) {

	unsigned int row  = get_global_id(0)>>3;
	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%8;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*8+idb];
			indcol = col[colStart[k]+row*8+idb];
			svalue += value * rhsVec[indcol];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELR8kernel (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen, local double *shared) {
	unsigned int row  = get_global_id(0)>>3;

	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%8;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*8 + 8*row + idb]; 
			indcol = col[k*pad*8 + 8*row + idb]; 
			svalue += value * rhsVec[indcol];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void pJDS16kernel (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart, local double *shared) {

	unsigned int row  = get_global_id(0)>>4;
	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%16;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*16+idb];
			indcol = col[colStart[k]+row*16+idb];
			svalue += value * rhsVec[indcol];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<8)
			shared[get_local_id(0)]+=shared[get_local_id(0)+8];
		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELR16kernel (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen, local double *shared) {
	unsigned int row  = get_global_id(0)>>4;

	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%16;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*16 + 16*row + idb]; 
			indcol = col[k*pad*16 + 16*row + idb]; 
			svalue += value * rhsVec[indcol];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<8)
			shared[get_local_id(0)]+=shared[get_local_id(0)+8];
		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void pJDS1kernelAdd (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart) {

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

kernel void ELR1kernelAdd (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen) {

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

kernel void pJDS2kernelAdd (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart, local double *shared) {

	unsigned int row  = get_global_id(0)>>1;
	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%2;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*2+idb];
			indcol = col[colStart[k]+row*2+idb];
			svalue += value * rhsVec[indcol];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELR2kernelAdd (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen, local double *shared) {
	unsigned int row  = get_global_id(0)>>1;

	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%2;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*2 + 2*row + idb]; 
			indcol = col[k*pad*2 + 2*row + idb]; 
			svalue += value * rhsVec[indcol];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void pJDS4kernelAdd (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart, local double *shared) {

	unsigned int row  = get_global_id(0)>>2;
	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%4;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*4+idb];
			indcol = col[colStart[k]+row*4+idb];
			svalue += value * rhsVec[indcol];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELR4kernelAdd (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen, local double *shared) {
	unsigned int row  = get_global_id(0)>>2;

	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%4;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*4 + 4*row + idb]; 
			indcol = col[k*pad*4 + 4*row + idb]; 
			svalue += value * rhsVec[indcol];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void pJDS8kernelAdd (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart, local double *shared) {

	unsigned int row  = get_global_id(0)>>3;
	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%8;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*8+idb];
			indcol = col[colStart[k]+row*8+idb];
			svalue += value * rhsVec[indcol];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELR8kernelAdd (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen, local double *shared) {
	unsigned int row  = get_global_id(0)>>3;

	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%8;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*8 + 8*row + idb]; 
			indcol = col[k*pad*8 + 8*row + idb]; 
			svalue += value * rhsVec[indcol];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void pJDS16kernelAdd (global double *resVec, global double *rhsVec, int nRows, global double *val, global int *col, global int *rowLen, global int *colStart, local double *shared) {

	unsigned int row  = get_global_id(0)>>4;
	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%16;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*16+idb];
			indcol = col[colStart[k]+row*16+idb];
			svalue += value * rhsVec[indcol];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<8)
			shared[get_local_id(0)]+=shared[get_local_id(0)+8];
		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELR16kernelAdd (global double *resVec, global double *rhsVec, int nRows, int pad, global double *val, global int *col, global int *rowLen, local double *shared) {
	unsigned int row  = get_global_id(0)>>4;

	if (row < nRows) {
		unsigned int indcol;
		unsigned short idb, k;
		double svalue, value;

		idb  = get_local_id(0)%16;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*16 + 16*row + idb]; 
			indcol = col[k*pad*16 + 16*row + idb]; 
			svalue += value * rhsVec[indcol];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb<8)
			shared[get_local_id(0)]+=shared[get_local_id(0)+8];
		if (idb<4)
			shared[get_local_id(0)]+=shared[get_local_id(0)+4];
		if (idb<2)
			shared[get_local_id(0)]+=shared[get_local_id(0)+2];

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void axpyKernel(global double *a, global double *b, double s, int nRows){
	int i = get_global_id(0); 
	if (i<nRows)
		a[i] += s*b[i]; 
}

kernel void vecscalKernel(global double *a, double scal, int nRows){
	int i = get_global_id(0);
	if (i<nRows)	
		a[i] *= scal; 
} 

kernel void dotprodKernel(global double *a, global double *b, global double *out, unsigned int nRows, local volatile double *shared) {

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
