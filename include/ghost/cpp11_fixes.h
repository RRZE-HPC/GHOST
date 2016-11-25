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

template<typename T> 
static inline T conj(const T& a)
{
    return std::conj(a);
}

#if __cplusplus < 201103L
static inline float conj(const float &a)
{
    return a;
}

static inline double conj(const double &a)
{
    return a;
}
#endif

template<typename T> 
static inline T norm(const T& a)
{
    return std::norm(a);
}

#if __cplusplus < 201103L
static inline float norm(const float &a)
{
    return std::fabs(a);
}

static inline double norm(const double &a)
{
    return std::fabs(a);
}
#endif

template<typename T>
static inline std::string to_string(T value)
{
    return std::to_string(value);
}

#if __cplusplus < 201103L
template<typename T>
static inline std::string to_string(T value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}
#endif


#endif
