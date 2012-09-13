#!/usr/bin/python

# show sparsity pattern of matrix market file

import sys
import numpy
import matplotlib.pyplot as pyplot
import scipy.sparse as sparse
from scipy.io.mmio import mmread

matrixname = sys.argv[1]
mode = int(sys.argv[2]) # 0=matrix, 1=perm, 2=colperm

fig = pyplot.figure()
spyfig = fig.add_subplot(111)

print("reading matrix")
matrix = mmread("/home/vault/unrz/unrza317/matrices/"+matrixname+"/"+matrixname+"_alt.mtx").tolil()
print("done")

if mode!=0:
	N = matrix.get_shape()[0]
	lens = []

	print("reading row lenghts")
	for row in matrix.rows:
		lens.append(len(row))
	print("done")



	print("inverse permutation vector")
	invpermvec = sorted(range(len(lens)),key=lens.__getitem__, reverse=True)
	print("done")


	print("permutation vector")
	permvec=N*[0]
	for i in range(N):
		permvec[invpermvec[i]]=i
	print("done")


	print("assembling sorted matrix")


	rows=[]
	data=[]

	for i in xrange(N):
		row = matrix.rows[invpermvec[i]]
		dat = matrix.data[invpermvec[i]]
		if mode==2:
			for j in xrange(len(row)):
				row[j] = permvec[row[j]]
		rows.append(row)
		data.append(dat)

	matrix.rows = rows
	matrix.data = data
	print("done")


spyfig.spy(matrix, marker=',', color='black')
spyfig.get_xaxis().set_visible(False)
spyfig.get_yaxis().set_visible(False)
if mode==0:
	pyplot.savefig(matrixname+".png",dpi=1500)
elif mode==1:
	pyplot.savefig(matrixname+"_perm.png",dpi=1500)
elif mode==2:
	pyplot.savefig(matrixname+"_colperm.png",dpi=1500)






