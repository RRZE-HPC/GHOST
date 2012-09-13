#!/bin/bash

# convert a single CRS matrix file to three files, containing the row pointers,
# column indices and values

MAT=dlr1
N=278502
NENTS=40025628

echo mat: "$MAT"

head -$((N+1)) ${MAT}.crs > ${MAT}_rowptr.crs ; tail -n +$((N+2)) ${MAT}.crs > ${MAT}_colval.crs
split -l $NENTS ${MAT}_colval.crs $MAT
mv ${MAT}aa ${MAT}_col.crs
mv ${MAT}ab ${MAT}_val.crs

