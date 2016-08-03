#!/bin/bash

grep -q -F 'x86intrin' $1 || sed -i '1s/^/\#ifdef __GNUC__\n\#include <x86intrin.h>\n\#endif\n/' $1

