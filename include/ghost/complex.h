#ifndef GHOST_COMPLEX_H
#define GHOST_COMPLEX_H

#ifdef __cplusplus

template <typename T> 
struct ghost_complex
{
        T re, im;
        ghost_complex() {re = 0.; im = 0.;}
        ghost_complex(T a)  {re = a; im = 0.;}
        ghost_complex(T a, T b) { re = a; im = b;}
        //ghost_complex(const std::complex<T> &a) : std::complex<T>(a) {re = a.real();}
        ghost_complex(const ghost_complex<float> &a) {re = a.re; im = a.im;}
        operator T() const;
        ghost_complex<T> operator-(const ghost_complex<T>&) const;
        ghost_complex<T> operator+(const ghost_complex<T>&) const;
        ghost_complex<T>& operator+=(const ghost_complex<T>&);
        ghost_complex<T> operator*(const ghost_complex<T>&) const;
        ghost_complex<T>& operator*=(const ghost_complex<T>&);
};

template <typename T>
ghost_complex<T>::operator T() const {
    return re;
}

template <typename T>
ghost_complex<T>& ghost_complex<T>::operator +=(const ghost_complex<T>& c) {
    *this = *this + c;
    return *this;
}

template <typename T>
ghost_complex<T>& ghost_complex<T>::operator *=(const ghost_complex<T>& c) {
    *this = *this * c;
    return *this;
}

template <typename T>
ghost_complex<T> ghost_complex<T>::operator +(const ghost_complex<T>& c) const {
    return ghost_complex<T>(this->re + c.re, this->im + c.im);
}

template <typename T>
ghost_complex<T> ghost_complex<T>::operator -(const ghost_complex<T>& c) const {
    return ghost_complex<T>(this->re - c.re, this->im - c.im);
}

template <typename T>
ghost_complex<T> ghost_complex<T>::operator *(const ghost_complex<T>& c) const {
    return ghost_complex<T>(this->re*c.re - this->im*c.im, this->re*c.im + this->im*c.re);
}


#endif
#endif
