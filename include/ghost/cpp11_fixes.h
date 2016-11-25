/**
 * @file cpp11_fixes.h
 * @brief Some smaller fixes to implement C++11 functionality for non-C++11 compilers.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CPP11_FIXES_H
#define GHOST_CPP11_FIXES_H

template<typename T> 
static inline T conj(const T& a)
{
    return conj(T);
}

static inline float conj(const float &a)
{
    return a;
}

static inline double conj(const double &a)
{
    return a;
}

template<typename T> 
static inline T norm(const T& a)
{
    return norm(T);
}

static inline float norm(const float &a)
{
    return std::fabs(a);
}

static inline double norm(const double &a)
{
    return std::fabs(a);
}

#endif
