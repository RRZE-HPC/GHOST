#!/bin/bash
# This script includes the missing header x86intrin.h if compiled with GCC

grep -q -F 'x86intrin' $1 || sed -i '1s/^/\#ifdef __GNUC__\n\#include <x86intrin.h>\n\#endif\n/' $1

