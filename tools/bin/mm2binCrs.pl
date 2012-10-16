#!/usr/bin/perl

use strict;
use warnings;
use Math::MatrixSparse;

my $mmMatrix = Math::MatrixSparse->newmatrixmarket($ARGV[0])->cull();
my $nRows = ($mmMatrix->dim())[0];
my $rowPtr = 0;
print pack('i',$rowPtr);

for (my $rowIdx = 1; $rowIdx<=$nRows; $rowIdx++) {
	$rowPtr += $mmMatrix->row($rowIdx)->sizetofit();
	print $rowPtr;
#	print pack('i',$rowPtr);
#print $row->elements(1,1)."\n";

}


