#!/apps/python/2.7.1/bin/python

import sys
import numpy
import scipy.sparse as sparse
import os.path
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

nrows = matrix.shape[0]
ncols = matrix.shape[1]
nnz = len(matrix.indices)
nc = nrows/chunksz

print os.path.splitext(os.path.basename(sys.argv[1]))[0],
print " & ",
print '{:,}'.format(nrows),
print " & ",
print '{:,}'.format(nnz),
print " & ",
print "%.2f" % (nnz*1./nrows),
print " & ",
print "%.2e" % (nnz*1./nrows/ncols),
print " & ",
	
rl = []
srl = []

for i in range(0, len(matrix.indptr)-1): # extract row lenghts
	rl.append(matrix.indptr[i+1]-matrix.indptr[i])

#srl = rl

#beta = 0
#nznc = nc

#for c in range(0,nc):
#	maxrowlen = 0
#	nnzs = 0
#
#	for i in range(c*chunksz,(c+1)*chunksz):
#		maxrowlen = max(srl[i],maxrowlen)
#		nnzs = nnzs + srl[i]
#
#	if maxrowlen == 0:
#		nznc = nznc-1
#		continue
#
#	beta = beta + (1.*nnzs/chunksz/maxrowlen)
#

# TODO: remainder
#beta = beta/nznc

#print "%.2f" % beta,
#print " & ",

srl = []
if sortsz > 0: # sort block-wise
	ns = nrows/sortsz

	for s in range(0,ns):
		srl = srl + sorted(rl[s*sortsz:(s+1)*sortsz])

	srl = srl + sorted(rl[(ns-1)*sortsz:]) # remainder

elif sortsz == 0: # don't sort
	srl = rl

else: # sort globally
	srl = sorted(rl)
	

beta = 0
nznc = nc

for c in range(0,nc):
	maxrowlen = 0
	nnzs = 0

	for i in range(c*chunksz,(c+1)*chunksz):
		maxrowlen = max(srl[i],maxrowlen)
		nnzs = nnzs + srl[i]


	if maxrowlen == 0:
		nznc = nznc-1
		continue

	beta = beta + (1.*nnzs/chunksz/maxrowlen)


# TODO: remainder

beta = beta/nznc

print "%.2f" % beta,

