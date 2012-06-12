
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



kernel void ELRkernel (int nRows, int pad, global double *resVec, global double *rhsVec, global double *val, global int *col, global int *rowLen) {

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

kernel void ELRkernelAdd (int nRows, int pad, global double *resVec, global double *rhsVec, global double *val, global int *col, global int *rowLen) {

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


kernel void pJDSkernel (int nRows, global double *resVec, global double *rhsVec, global double *val, global int *col, global int *rowLen, global int *colStart) {

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

kernel void pJDSkernelAdd (int nRows, global double *resVec, global double *rhsVec, global double *val, global int *col, global int *rowLen, global int *colStart) {

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
