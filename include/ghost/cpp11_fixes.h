/**
 * @file cpp11_fixes.h
 * @brief Some smaller fixes to implement C++11 functionality for non-C++11 compilers.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CPP11_FIXES_H
#define GHOST_CPP11_FIXES_H

#include <string>
#include <sstream>
#include <complex>

namespace ghost {

// Difference between ghost::conj_or_nop and std::conj
//  std::conj(double) -> std::complex<double>
//  ghost::conj_or_nop(double) -> double
template<typename T>
static inline T conj_or_nop(const T &a)
{
    return std::conj(a);
}

static inline float conj_or_nop(const float &a) { return a; }

static inline double conj_or_nop(const double &a) { return a; }

template<typename T>
static inline T norm(const T &a)
{
    return std::norm(a);
}

#if __cplusplus < 201103L
static inline float norm(const float &a)
{
    /*return std::fabs(a);*/
    return a * a;
}

static inline double norm(const double &a)
{
    /*return std::fabs(a);*/
    return a * a;
}
#endif

#if __cplusplus < 201103L
template<typename T>
static inline std::string to_string(T value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}
#else
template<typename T>
static inline std::string to_string(T value)
{
    return std::to_string(value);
}
#endif
}

#endif
