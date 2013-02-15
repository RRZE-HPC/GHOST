#!/usr/bin/python

import sys
import numpy
import scipy.sparse as sparse
from scipy.io.mmio import mmread,mminfo
from ctypes import *
import bisect

if len(sys.argv) != 3:
	print("Usage: "+sys.argv[0]+" matrixPath dataType");
	print("\tmatrixPath: path to Matrix Market file");
	print("\tdataType  : output data type, one of");
	print("\t\t 5 .. float");
	print("\t\t 6 .. double");
	print("\t\t 9 .. complex float");
	print("\t\t 10 .. complex double");
	sys.exit(0);

fileformatversion = 1;
endianess = 0;
base = 0;

outfile = open('out.crs','wb')

matrixpath = sys.argv[1]
datatype = int(sys.argv[2])
info = mminfo(matrixpath)

if info[5] == 'general':
	symm = 1
elif info[5] == 'symmetric':
	symm = 2
elif info[5] == 'skew-symmetric':
	symm = 4
elif info[6] == 'hermitian':
	symm = 8

symm = 1

outfile.write(c_int(endianess))
outfile.write(c_int(fileformatversion))
outfile.write(c_int(base))
outfile.write(c_int(symm))
outfile.write(c_int(datatype))
outfile.write(c_longlong(int(info[0])))
outfile.write(c_longlong(int(info[1])))
#outfile.write(c_longlong(int(info[2])))

rpt = 0

#if symm == 2:
#	matrix = mmread(matrixpath).tolil()
#	for r in range(0,int(info[0])):
#		outfile.write(c_longlong(rpt))
#		row = matrix.rows[r]
#		upperlen = len(row[bisect.bisect(row,r-1):])
#		rpt = rpt + upperlen
#	
#	outfile.write(c_longlong(rpt))
#
#	for r in range(0,int(info[0])):
#		row = matrix.rows[r]
#		for entry in row[bisect.bisect(row,r-1):]:
#			outfile.write(c_longlong(entry))
#	
#	for r in range(0,int(info[0])):
#		data = matrix.data[r]
#		row = matrix.rows[r]
#		for entry in data[bisect.bisect(row,r-1):]:
#			if datatype == 5:
#				outfile.write(c_float(entry.real))
#			elif datatype == 6:
#				outfile.write(c_double(entry.real))
#			elif datatype == 9:
#				outfile.write(c_float(entry.real))
#				outfile.write(c_float(entry.imag))
#			elif datatype == 10:
#				outfile.write(c_double(entry.real))
#				outfile.write(c_double(entry.imag))

#elif symm == 1:
matrix = mmread(matrixpath).tocsr()
outfile.write(c_longlong(matrix.nnz))

for entry in matrix.indptr:
	outfile.write(c_longlong(entry))

for entry in matrix.indices:
	outfile.write(c_longlong(entry))

if datatype == 5:
	for entry in matrix.data:
		outfile.write(c_float(entry.real))
elif datatype == 6:
	for entry in matrix.data:
		outfile.write(c_double(entry.real))
elif datatype == 9:
	for entry in matrix.data:
		outfile.write(c_float(entry.real))
		outfile.write(c_float(entry.imag))
elif datatype == 10:
	for entry in matrix.data:
		outfile.write(c_double(entry.real))
		outfile.write(c_double(entry.imag))

#else:
#	print "Can not handle this type of symmetry!"

