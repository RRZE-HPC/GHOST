#!/usr/bin/perl 

# convert three CRS matrix files to a special CRS matrix file, where the first
# n lines look like "$val, $col, $rowptr" and the following like "$val, $col"

use strict;
use warnings;


my $ROW_filename     = $ARGV[0];
my $COLUMN_filename  = $ARGV[1];
my $VALUE_filename   = $ARGV[2];

open COLFILE,"<$COLUMN_filename";
open ROWFILE,"<$ROW_filename";
open VALUEFILE,"<$VALUE_filename";

while (my $rowptr = <ROWFILE>) {
	chomp($rowptr);
	chomp(my $col =  <COLFILE>);
    chomp(my $val =  <VALUEFILE>);
	$col = $col-1;
	$rowptr = $rowptr-1;
	print "$val $col $rowptr\n";
}

while (my $val = <VALUEFILE>) {
	chomp($val);
	chomp(my $col =  <COLFILE>);
	$col = $col-1;
	print "$val $col\n";
}


close COLFILE;
close ROWFILE;
close VALUEFILE;

