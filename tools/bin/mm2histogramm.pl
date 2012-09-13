#!/usr/bin/perl 

use strict;
use warnings;


my $MATRIX_filename   = $ARGV[0];
my $header;

open MATRIXFILE,"<$MATRIX_filename";   # open mtx file
while (<MATRIXFILE>) {
	s/%.*//;                  # skip comments
	next if /^(\s)*$/;
	$header = $_;             # read header
	last;
}
$header =~ s/^\s+//;             # remove leading spaces

my @entry = split(/\s+/, $header);  # split header
my $nrows = $entry[0];
my @entriesPerRow = ((0) x $nrows); # initialize entries per row to zero for each row

# read entry by entry
while (my $line =  <MATRIXFILE>) {

	$line =~ s/^\s+//;             # remove leading spaces
	@entry = split(/\s+/, $line);  # split line
	$entriesPerRow[$entry[0]-1]++; # 0th entry is the row idx, "-1": file is 1-based
}

my %rowlen;

foreach (@entriesPerRow) {
	if (defined $rowlen{$_}) {
		$rowlen{$_}++;
	} else {
		$rowlen{$_} = 1;
	}
}

my @sortedKeys = sort {$a<=>$b} keys %rowlen;


#print "# len nRows\n";
#for (my $l = 1; $l<=((@sortedKeys)[-1]); $l++) {
#	if (defined $rowlen{$l}) {
#		print "$l $rowlen{$l}\n";
#	} else {
#		print "$l 0\n";
#	}
#}

print "# len nRows\n";
for (my $l = 1; $l<=((@sortedKeys)[-1]); $l++) {
	if (defined $rowlen{$l}) {
		print $l." ".$rowlen{$l}/$nrows."\n";
	} else {
		print "$l 0\n";
	}
}

close MATRIXFILE;

