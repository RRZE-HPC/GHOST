#ifndef GHOST_CU_COMPLEX_H
#define GHOST_CU_COMPLEX_H

template<typename T>
__device__ inline void zero(T &val)
{
    val = 0.;
}

template<>
__device__ inline void zero<cuFloatComplex>(cuFloatComplex &val)
{
    val = make_cuFloatComplex(0.,0.);
}

template<>
__device__ inline void zero<cuDoubleComplex>(cuDoubleComplex &val)
{
    val = make_cuDoubleComplex(0.,0.);
}

template<typename T>
__device__ inline void one(T &val)
{
    val = 1.;
}

template<>
__device__ inline void one<cuFloatComplex>(cuFloatComplex &val)
{
    val = make_cuFloatComplex(1.,1.);
}

template<>
__device__ inline void one<cuDoubleComplex>(cuDoubleComplex &val)
{
    val = make_cuDoubleComplex(1.,1.);
}

// val += val2*val3
template<typename T, typename T2>
__device__ inline T axpy(T val, T val2, T2 val3)
{
    return val+val2*val3;
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,cuFloatComplex>(cuFloatComplex val, cuFloatComplex val2, cuFloatComplex val3)
{
    return cuCaddf(val,cuCmulf(val2,val3));
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,double>(cuFloatComplex val, cuFloatComplex val2, double val3)
{
    return cuCaddf(val,cuCmulf(val2,make_cuFloatComplex((float)val3,0.f)));
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,float>(cuFloatComplex val, cuFloatComplex val2, float val3)
{
    return cuCaddf(val,cuCmulf(val2,make_cuFloatComplex(val3,0.f)));
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,cuDoubleComplex>(cuFloatComplex val, cuFloatComplex val2, cuDoubleComplex val3)
{
    return cuCaddf(val,cuCmulf(val2,make_cuFloatComplex((float)(cuCreal(val3)),(float)(cuCimag(val3)))));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,double>(cuDoubleComplex val, cuDoubleComplex val2, double val3)
{
    return cuCadd(val,cuCmul(val2,make_cuDoubleComplex(val3,0.)));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,float>(cuDoubleComplex val, cuDoubleComplex val2, float val3)
{
    return cuCadd(val,cuCmul(val2,make_cuDoubleComplex((double)val3,0.)));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,cuDoubleComplex>(cuDoubleComplex val, cuDoubleComplex val2, cuDoubleComplex val3)
{
    return cuCadd(val,cuCmul(val2,val3));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,cuFloatComplex>(cuDoubleComplex val, cuDoubleComplex val2, cuFloatComplex val3)
{
    return cuCadd(val,cuCmul(val2,make_cuDoubleComplex((double)(cuCrealf(val3)),(double)(cuCimagf(val3)))));
}

template<>
__device__ inline double axpy<double,cuFloatComplex>(double val, double val2, cuFloatComplex val3)
{
    return val+val2*(double)cuCrealf(val3);
}

template<>
__device__ inline double axpy<double,cuDoubleComplex>(double val, double val2, cuDoubleComplex val3)
{
    return val+val2*cuCreal(val3);
}

template<>
__device__ inline float axpy<float,cuFloatComplex>(float val, float val2, cuFloatComplex val3)
{
    return val+val2*cuCrealf(val3);
}


template<>
__device__ inline float axpy<float,cuDoubleComplex>(float val, float val2, cuDoubleComplex val3)
{
    return val+val2*(float)cuCreal(val3);
}

// y = a*x + b*y
template<typename T>
__device__ inline T axpby(T x, T y, T a, T b)
{
    return b*y+a*x;
}

template<>
__device__ inline cuFloatComplex axpby<cuFloatComplex>(cuFloatComplex x, cuFloatComplex y, cuFloatComplex a, cuFloatComplex b)
{
    return cuCaddf(cuCmulf(b,y),cuCmulf(a,x));
}

template<>
__device__ inline cuDoubleComplex axpby<cuDoubleComplex>(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex a, cuDoubleComplex b)
{
    return cuCadd(cuCmul(b,y),cuCmul(a,x));
}

// x = a*y
template<typename T>
__device__ inline T scale(T y, T a)
{
    return a*y;
}

template<>
__device__ inline cuFloatComplex scale<cuFloatComplex>(cuFloatComplex y, cuFloatComplex a)
{
    return cuCmulf(a,y);
}

template<>
__device__ inline cuDoubleComplex scale<cuDoubleComplex>(cuDoubleComplex y, cuDoubleComplex a)
{
    return cuCmul(a,y);
}

template<typename T1, typename T2>
__device__ inline T1 scale2(T1 y, T2 a)
{
    return a*y;
}

template<>
__device__ inline cuFloatComplex scale2<cuFloatComplex,cuFloatComplex>(cuFloatComplex y, cuFloatComplex a)
{
    return cuCmulf(a,y);
}

template<>
__device__ inline cuFloatComplex scale2<cuFloatComplex,float>(cuFloatComplex y, float a)
{
    return cuCmulf(make_cuFloatComplex(a,a),y);
}

template<>
__device__ inline cuFloatComplex scale2<cuFloatComplex,double>(cuFloatComplex y, double a)
{
    return cuCmulf(make_cuFloatComplex(a,a),y);
}

template<>
__device__ inline cuDoubleComplex scale2<cuDoubleComplex,cuDoubleComplex>(cuDoubleComplex y, cuDoubleComplex a)
{
    return cuCmul(a,y);
}


template<>
__device__ inline cuDoubleComplex scale2<cuDoubleComplex,float>(cuDoubleComplex y, float a)
{
    return cuCmul(make_cuDoubleComplex(a,a),y);
}

template<>
__device__ inline cuDoubleComplex scale2<cuDoubleComplex,double>(cuDoubleComplex y, double a)
{
    return cuCmul(make_cuDoubleComplex(a,a),y);
}


#endif
