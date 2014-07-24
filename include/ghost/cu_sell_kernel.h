#ifndef GHOST_CU_SELL_KERNEL_H
#define GHOST_CU_SELL_KERNEL_H

extern __shared__ char shared[];

template<typename v_t>
__device__ inline
v_t shfl_down(v_t var, unsigned int srcLane) {
    return __shfl_down(var, srcLane, warpSize);
}

template<>
__device__ inline
double shfl_down<double>(double var, unsigned int srcLane) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, warpSize);
    a.y = __shfl_down(a.y, srcLane, warpSize);
    return *reinterpret_cast<double*>(&a);
}

template<>
__device__ inline
cuFloatComplex shfl_down<cuFloatComplex>(cuFloatComplex var, unsigned int srcLane) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, warpSize);
    a.y = __shfl_down(a.y, srcLane, warpSize);
    return *reinterpret_cast<cuFloatComplex*>(&a);
}

template<>
__device__ inline
cuDoubleComplex shfl_down<cuDoubleComplex>(cuDoubleComplex var, unsigned int srcLane) {
    int4 a = *reinterpret_cast<int4*>(&var);
    a.x = __shfl_down(a.x, srcLane, warpSize);
    a.y = __shfl_down(a.y, srcLane, warpSize);
    a.z = __shfl_down(a.z, srcLane, warpSize);
    a.w = __shfl_down(a.w, srcLane, warpSize);
    return *reinterpret_cast<cuDoubleComplex*>(&a);
}

template<typename v_t>
__inline__ __device__
v_t warpReduceSum(v_t val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) { 
        val = axpy<v_t>(val,shfl_down(val, offset),1.f);
    }
    return val;
}

template<typename v_t>
__inline__ __device__
v_t blockReduceSum(v_t val) {

    v_t * shmem = (v_t *)shared; // Shared mem for 32 partial sums

    int lane = (threadIdx.x % warpSize) + (32*threadIdx.y);
    int wid = (threadIdx.x / warpSize) + (32*threadIdx.y);

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (threadIdx.x%warpSize == 0) shmem[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    if (threadIdx.x < blockDim.x / warpSize) {
        val = shmem[lane];
    } else {
        zero<v_t>(val);
    }

    if (threadIdx.x/warpSize == 0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

template<typename v_t>
__global__ void deviceReduceKernel(v_t *in, v_t* out, int N) {
    v_t sum;
    zero<v_t>(sum);
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < N; 
            i += blockDim.x * gridDim.x) {
        sum = axpy<v_t>(sum,in[i],1.f);
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x==0)
        out[blockIdx.x]=sum;
}



#endif
