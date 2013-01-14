#!/usr/bin/perl 

# convert three CRS files to a matrix market file

use strict;
use warnings;


my $ROW_filename     = $ARGV[0];
my $COLUMN_filename  = $ARGV[1];
my $VALUE_filename   = $ARGV[2];

open COLFILE,"<$COLUMN_filename";
open ROWFILE,"<$ROW_filename";
open VALUEFILE,"<$VALUE_filename";

chomp(my $row0 = <ROWFILE>);
my $rowID = 1;

while (chomp (my $row1 = <ROWFILE>)) {

    my @colVal;
    my @Val;

    foreach my $id ($row0 .. $row1-1) {
        chomp(my $col =  <COLFILE>);
        chomp(my $val =  <VALUEFILE>);
		print "$rowID $col $val\n";

    }

    $rowID++;
    $row0 = $row1;
#    last if ($rowID == 30);
}

close COLFILE;
close ROWFILE;
close VALUEFILE;

