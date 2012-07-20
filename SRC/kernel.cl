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

#ifdef DOUBLE
#ifdef COMPLEX
typedef double2 clreal;
#else
typedef double clreal;
#endif
#endif
#ifdef SINGLE
#ifdef COMPLEX
typedef float2 clreal;
#else
typedef float clreal;
#endif
#endif

kernel void pJDS1kernel (global clreal *resVec, global clreal *rhsVec, int nRows, global clreal *val, global int *col, global int *rowLen, global int *colStart) {

	int row = get_global_id(0);
	clreal svalue = 0.0, value, rhs;
	int i, idcol;

	if (row < nRows) {
		for( i = 0; i < rowLen[row]; ++i) {
			value = val[colStart[i]+row];
			idcol = col[colStart[i]+row];
			rhs = rhsVec[idcol];
#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
		}
		resVec[row] = svalue;
	}

}

kernel void ELR1kernel (global clreal *resVec, global clreal *rhsVec, int nRows, int pad, global clreal *val, global int *col, global int *rowLen) {

	int row = get_global_id(0);
	clreal svalue = 0.0, value, rhs;
	int i, idcol;
	if (row < nRows) {


		for( i = 0; i < rowLen[row]; ++i) {
			value = val[i*pad+row];
			idcol = col[i*pad+row];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
		}
		resVec[row] = svalue;
	}
}

kernel void pJDS2kernel (global clreal *resVec, global clreal *rhsVec, int nRows, global clreal *val, global int *col, global int *rowLen, global int *colStart, local clreal *shared) {

	unsigned int row  = get_global_id(0)>>1;
	if (row < nRows) {
			unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%2;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*2+idb];
			idcol = col[colStart[k]+row*2+idb];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELR2kernel (global clreal *resVec, global clreal *rhsVec, int nRows, int pad, global clreal *val, global int *col, global int *rowLen, local clreal *shared) {
	unsigned int row  = get_global_id(0)>>1;

	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%2;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*2 + 2*row + idb]; 
			idcol = col[k*pad*2 + 2*row + idb]; 
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb==0) {
			resVec[row] = shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void pJDS4kernel (global clreal *resVec, global clreal *rhsVec, int nRows, global clreal *val, global int *col, global int *rowLen, global int *colStart, local clreal *shared) {

	unsigned int row  = get_global_id(0)>>2;
	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%4;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*4+idb];
			idcol = col[colStart[k]+row*4+idb];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void ELR4kernel (global clreal *resVec, global clreal *rhsVec, int nRows, int pad, global clreal *val, global int *col, global int *rowLen, local clreal *shared) {
	unsigned int row  = get_global_id(0)>>2;

	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%4;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*4 + 4*row + idb]; 
			idcol = col[k*pad*4 + 4*row + idb]; 
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void pJDS8kernel (global clreal *resVec, global clreal *rhsVec, int nRows, global clreal *val, global int *col, global int *rowLen, global int *colStart, local clreal *shared) {

	unsigned int row  = get_global_id(0)>>3;
	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%8;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*8+idb];
			idcol = col[colStart[k]+row*8+idb];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void ELR8kernel (global clreal *resVec, global clreal *rhsVec, int nRows, int pad, global clreal *val, global int *col, global int *rowLen, local clreal *shared) {
	unsigned int row  = get_global_id(0)>>3;

	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%8;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*8 + 8*row + idb]; 
			idcol = col[k*pad*8 + 8*row + idb]; 
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void pJDS16kernel (global clreal *resVec, global clreal *rhsVec, int nRows, global clreal *val, global int *col, global int *rowLen, global int *colStart, local clreal *shared) {

	unsigned int row  = get_global_id(0)>>4;
	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%16;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*16+idb];
			idcol = col[colStart[k]+row*16+idb];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void ELR16kernel (global clreal *resVec, global clreal *rhsVec, int nRows, int pad, global clreal *val, global int *col, global int *rowLen, local clreal *shared) {
	unsigned int row  = get_global_id(0)>>4;

	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%16;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*16 + 16*row + idb]; 
			idcol = col[k*pad*16 + 16*row + idb]; 
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void pJDS1kernelAdd (global clreal *resVec, global clreal *rhsVec, int nRows, global clreal *val, global int *col, global int *rowLen, global int *colStart) {

	int row = get_global_id(0);
	clreal svalue = 0.0, value, rhs;
	int i, idcol;

	if (row < nRows) {
		for( i = 0; i < rowLen[row]; ++i) {
			value = val[colStart[i]+row];
			idcol = col[colStart[i]+row];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
		}
		resVec[row] += svalue;
	}

}

kernel void ELR1kernelAdd (global clreal *resVec, global clreal *rhsVec, int nRows, int pad, global clreal *val, global int *col, global int *rowLen) {

	int row = get_global_id(0);
	clreal svalue = 0.0, value, rhs;
	int i, idcol;
	if (row < nRows) {


		for( i = 0; i < rowLen[row]; ++i) {
			value = val[i*pad+row];
			idcol = col[i*pad+row];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
		}
		resVec[row] += svalue;

	}
}

kernel void pJDS2kernelAdd (global clreal *resVec, global clreal *rhsVec, int nRows, global clreal *val, global int *col, global int *rowLen, global int *colStart, local clreal *shared) {

	unsigned int row  = get_global_id(0)>>1;
	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%2;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*2+idb];
			idcol = col[colStart[k]+row*2+idb];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
} 

kernel void ELR2kernelAdd (global clreal *resVec, global clreal *rhsVec, int nRows, int pad, global clreal *val, global int *col, global int *rowLen, local clreal *shared) {
	unsigned int row  = get_global_id(0)>>1;

	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%2;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*2 + 2*row + idb]; 
			idcol = col[k*pad*2 + 2*row + idb]; 
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
		} 
		barrier(CLK_LOCAL_MEM_FENCE);

		shared[get_local_id(0)] = svalue;

		if (idb==0) {
			resVec[row] += shared[get_local_id(0)]+shared[get_local_id(0)+1];
		}
	}
}

kernel void pJDS4kernelAdd (global clreal *resVec, global clreal *rhsVec, int nRows, global clreal *val, global int *col, global int *rowLen, global int *colStart, local clreal *shared) {

	unsigned int row  = get_global_id(0)>>2;
	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%4;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*4+idb];
			idcol = col[colStart[k]+row*4+idb];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void ELR4kernelAdd (global clreal *resVec, global clreal *rhsVec, int nRows, int pad, global clreal *val, global int *col, global int *rowLen, local clreal *shared) {
	unsigned int row  = get_global_id(0)>>2;

	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%4;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*4 + 4*row + idb]; 
			idcol = col[k*pad*4 + 4*row + idb]; 
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void pJDS8kernelAdd (global clreal *resVec, global clreal *rhsVec, int nRows, global clreal *val, global int *col, global int *rowLen, global int *colStart, local clreal *shared) {

	unsigned int row  = get_global_id(0)>>3;
	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%8;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*8+idb];
			idcol = col[colStart[k]+row*8+idb];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void ELR8kernelAdd (global clreal *resVec, global clreal *rhsVec, int nRows, int pad, global clreal *val, global int *col, global int *rowLen, local clreal *shared) {
	unsigned int row  = get_global_id(0)>>3;

	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%8;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*8 + 8*row + idb]; 
			idcol = col[k*pad*8 + 8*row + idb]; 
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void pJDS16kernelAdd (global clreal *resVec, global clreal *rhsVec, int nRows, global clreal *val, global int *col, global int *rowLen, global int *colStart, local clreal *shared) {

	unsigned int row  = get_global_id(0)>>4;
	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%16;

		svalue = 0.0;
		for( k = 0; k < rowLen[row]; ++k)
		{
			value = val[colStart[k]+row*16+idb];
			idcol = col[colStart[k]+row*16+idb];
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void ELR16kernelAdd (global clreal *resVec, global clreal *rhsVec, int nRows, int pad, global clreal *val, global int *col, global int *rowLen, local clreal *shared) {
	unsigned int row  = get_global_id(0)>>4;

	if (row < nRows) {
		unsigned int idcol;
		unsigned short idb, k;
		clreal svalue, value, rhs;

		idb  = get_local_id(0)%16;
		svalue = 0.0;
		for(k=0; k<rowLen[row]; ++k){ 

			value = val[k*pad*16 + 16*row + idb]; 
			idcol = col[k*pad*16 + 16*row + idb]; 
			rhs = rhsVec[idcol];

#ifdef COMPLEX
			svalue.s0 += (value.s0*rhs.s0 - value.s1*rhs.s1);
			svalue.s1 += (value.s0*rhs.s1 + value.s1*rhs.s0);
#else
			svalue += value*rhs;
#endif
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

kernel void axpyKernel(global clreal *a, global clreal *b, clreal s, int nRows){
	int i = get_global_id(0); 
	if (i<nRows)
		a[i] += s*b[i]; 
}

kernel void vecscalKernel(global clreal *a, clreal scal, int nRows){
	int i = get_global_id(0);
	if (i<nRows)	
		a[i] *= scal; 
} 

// TODO kernels with complex support

kernel void dotprodKernel(global clreal *a, global clreal *b, global clreal *out, unsigned int nRows, local volatile clreal *shared) {

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
