#!/usr/bin/perl

# convert matrix market file to CRS files

use strict;
use warnings;

my $matrixname = $ARGV[0];

my $mtx_filename = $matrixname.'/'.$matrixname.'.mtx';

my $rowptr_filename = $matrixname.'_rowptr.crs';
my $col_filename = $matrixname.'_col.crs';
my $val_filename = $matrixname.'_val.crs';

open MTXFILE,"<$mtx_filename";
open RPTFILE,">$rowptr_filename";
open COLFILE,">$col_filename";
open VALFILE,">$val_filename";


my $prevRow = 1;
my $rowPtr = 1;
my $rowLen = 0;


print RPTFILE "$rowPtr\n";

my $entry = <MTXFILE>;



while (my $entry = <MTXFILE>) {
	my @entryfields = split(' ',$entry);
	my $row = $entryfields[0];
	my $col = $entryfields[1];
	my $val = $entryfields[2];

	print COLFILE "$col\n";
	print VALFILE "$val\n";

	$rowLen++;
	if ($row != $prevRow) {
		$rowPtr+=$rowLen;
		print RPTFILE "$rowPtr\n"; 
		$prevRow = $row;
		$rowLen = 0;
	}
	

}

close MTXFILE;
close RPTFILE;
close COLFILE;
close VALFILE;
