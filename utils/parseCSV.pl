#!/usr/bin/perl

use strict;
use warnings;
use Text::CSV;

my %energies;
my $regionName;

foreach (@ARGV) {
	my $csv = Text::CSV->new();

	open (CSV,"<",$_) or die $!;

	while (<CSV>) {
		if ($csv->parse($_)) {
			my @columns = $csv->fields();

			if (index($columns[0],"Region:") != -1) {        # region name is in this line
				my @tokens = split(/ /,$columns[0]);         # extract region name
				$regionName = "@tokens[1..$#tokens]";
			}
			if ($columns[0] eq "Energy [J] STAT") {          # energy of region is in this line
				$energies{ $regionName } += "$columns[1]";   # add to total energy of region
			}
		} 


	}

	close CSV;
}

foreach my $k (sort keys %energies) {
	    print "$k: $energies{$k} J\n";
}
