/**
 * @file complex.h
 * @brief Header file of GHOSTS's complex number implementation.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_COMPLEX_H
#define GHOST_COMPLEX_H

#ifdef __cplusplus

#include <ostream>

template <typename T> 
struct ghost_complex
{
        T re, im;
        ghost_complex() {re = 0.; im = 0.;}
        ghost_complex(T a)  {re = a; im = 0.;}
        ghost_complex(T a, T b) { re = a; im = b;}
        //ghost_complex(const std::complex<T> &a) : std::complex<T>(a) {re = a.real();}
        ghost_complex(const ghost_complex<float> &a) {re = a.re; im = a.im;}
        operator T();
        ghost_complex<T> operator-(const ghost_complex<T>&);
        ghost_complex<T> operator+(const ghost_complex<T>&);
        ghost_complex<T>& operator+=(const ghost_complex<T>&);
        ghost_complex<T> operator*(const ghost_complex<T>&);
        ghost_complex<T>& operator*=(const ghost_complex<T>&);
        ghost_complex<T> operator/(const ghost_complex<T>&);
        ghost_complex<T>& operator/=(const ghost_complex<T>&);
        std::ostream& operator <<(std::ostream& o);
};

template <typename T>
ghost_complex<T>::operator T() {
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
ghost_complex<T>& ghost_complex<T>::operator /=(const ghost_complex<T>& c) {
    *this = *this / c;
    return *this;
}

template <typename T>
ghost_complex<T> ghost_complex<T>::operator +(const ghost_complex<T>& c) {
    return ghost_complex<T>(this->re + c.re, this->im + c.im);
}

template <typename T>
ghost_complex<T> ghost_complex<T>::operator -(const ghost_complex<T>& c) {
    return ghost_complex<T>(this->re - c.re, this->im - c.im);
}

template <typename T>
ghost_complex<T> ghost_complex<T>::operator *(const ghost_complex<T>& c) {
    return ghost_complex<T>(this->re*c.re - this->im*c.im, this->re*c.im + this->im*c.re);
}

template <typename T>
ghost_complex<T> ghost_complex<T>::operator /(const ghost_complex<T>& c) {
    T scale = c.re*c.re + c.im*c.im;
    return ghost_complex<T>((this->re*c.re + this->im*c.im)/scale, (this->im*c.re - this->re*c.im)/scale);
}

template <typename T>
std::ostream& ghost_complex<T>::operator <<(std::ostream& o) {
    o << "(" << this->re << ", " << this->im << ")";
    return o;
}




#endif
#endif
