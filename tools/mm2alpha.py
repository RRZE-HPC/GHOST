#!/usr/bin/python

import sys
import numpy
import scipy.sparse as sparse
from scipy.io.mmio import mmread,mminfo
from ctypes import *
import bisect

if len(sys.argv) != 4:
	print("Usage: "+sys.argv[0]+" matrixPath chunkSize");
	print("\tmatrixPath: path to Matrix Market file");
	print("\tchunkSize : the chunk size (e.g., SIMD size)");
	print("\tsortSize  : the sort block size");
	sys.exit(0);


matrix = mmread(sys.argv[1]).tocsr()
chunksz = int(sys.argv[2])
sortsz = int(sys.argv[3])

nrows = len(matrix.indptr)-1
nnz = len(matrix.indices)
nc = nrows/chunksz

print "nrows  : %d" % nrows
print "nnz    : %d" % nnz
print "nnz/row: %.2f" % (nnz*1./nrows)
print "chunksz: %d" % chunksz
print "sortsz : %d" % sortsz
	
rl = []
srl = []

for i in range(0, len(matrix.indptr)-1): # extract row lenghts
	rl.append(matrix.indptr[i+1]-matrix.indptr[i])


if sortsz > 0: # sort block-wise
	ns = nrows/sortsz

	for s in range(0,ns):
		srl = srl + sorted(rl[s*sortsz:(s+1)*sortsz])

	srl = srl + sorted(rl[(ns-1)*sortsz:]) # remainder

elif sortsz == 0: # don't sort
	srl = rl

else: # sort globally
	srl = sorted(rl)
	

alpha = 0

for c in range(0,nc):
	maxrowlen = 0
	for i in range(c*chunksz,(c+1)*chunksz):
		maxrowlen = max(srl[i],maxrowlen)
	
	fillin = 0
	for i in range(c*chunksz,(c+1)*chunksz):
		fillin = fillin + maxrowlen - srl[i]

	alpha = alpha + (1 - fillin*1./(chunksz-1)/maxrowlen)


# TODO: remainder

alpha = alpha/nc

print "============="
print "alpha  : %.2f" % alpha

