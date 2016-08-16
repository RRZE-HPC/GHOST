#!/usr/bin/env perl

use strict;
use warnings;
use integer;

my %substitutions;
while (<>) {
    if (/#GHOST_SUBST/) {
        my @split = split / /,$_;
        $substitutions{$split[1]} = $split[2];
    } else {
        while ( my ($key, $value) = each(%substitutions) ) {
            $_ =~ s/$key/$value/gee;
        }
        $_ =~ s/~([()\d\-\+\*\/%]+)~/$1/gee; # evaulate constant expresssion between "~" markers (usually used to construct variable names)
        if ($_ =~ /#GHOST_UNROLL/) {
            unroll($_);
        } 
        print $_;
    } 
}

sub unroll {
    my $spaces =  (split /#/,$_[0])[0];
    my $codeline =  (split /#/,$_[0])[2];
    my $unrollsize = (split /#/,$_[0])[3];
    $unrollsize =~ s/\\//g; # delete backslash from unrollsize (happens in macros)
    chomp($unrollsize);
    $unrollsize =~ s/([()\d\-\+\*\/%]+)/$1/gee; # evaluate unroll size 
    
    $_[0] = "";
    for (my $i=0; $i<$unrollsize; $i++) {
        my $modcodeline = $codeline;
        $modcodeline =~ s/@/$i/g;
        $modcodeline =~ s/~([()\d\-\+\*\/%]+)~/$1/gee; # evaulate constant expresssion between "~" markers (usually used to construct variable names)
        $_[0] = $_[0].$spaces.$modcodeline."\n";
    }
}
