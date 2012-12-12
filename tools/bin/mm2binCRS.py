#!/usr/bin/python

# show sparsity pattern of matrix market file

import sys
import numpy
import scipy.sparse as sparse
from scipy.io.mmio import mmread,mminfo
from ctypes import *

if len(sys.argv) != 3:
	print("Usage: "+sys.argv[0]+" matrixPath dataType");
	print("\tmatrixPath: path to Matrix Market file");
	print("\tdataType  : output data type, one of");
	print("\t\t 0 .. float");
	print("\t\t 1 .. double");
	print("\t\t 2 .. complex float");
	print("\t\t 3 .. complex double");
	sys.exit(0);

fileformatversion = 1

outfile = open('out.crs','wb')

matrixpath = sys.argv[1]
datatype = int(sys.argv[2])
matrix = mmread(matrixpath).tocsr()
info = mminfo(matrixpath)

if info[5] == 'general':
	symm = 0
elif info[5] == 'symmetric':
	symm = 1
elif info[5] == 'skew-symmetric':
	symm = 2
elif info[6] == 'hermitian':
	symm = 3

outfile.write(c_int(fileformatversion))
outfile.write(c_int(symm))
outfile.write(c_int(datatype))
outfile.write(c_longlong(int(info[0])))
outfile.write(c_longlong(int(info[1])))
outfile.write(c_longlong(int(info[2])))

for entry in matrix.indptr:
	outfile.write(c_longlong(entry))

for entry in matrix.indices:
	outfile.write(c_longlong(entry))

if datatype == 0:
	for entry in matrix.data:
		outfile.write(c_float(entry.real))
elif datatype == 1:
	for entry in matrix.data:
		outfile.write(c_double(entry.real))
elif datatype == 2:
	for entry in matrix.data:
		outfile.write(c_float(entry.real))
		outfile.write(c_float(entry.imag))
elif datatype == 3:
	for entry in matrix.data:
		outfile.write(c_double(entry.real))
		outfile.write(c_double(entry.imag))
