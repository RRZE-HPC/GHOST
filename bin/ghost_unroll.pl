#!/usr/bin/env perl

use strict;
use warnings;
use integer;

my $savelines = 0;
my $savedlines;
my %substitutions;
my $spaces;
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
            $spaces =  (split /#/,$_)[0];
            my $codeline =  (split /#/,$_)[2];
            my $unrollsize = (split /#/,$_)[3];
            $unrollsize =~ s/\\//g; # delete backslash from unrollsize (happens in macros)
            chomp($unrollsize);
            $unrollsize =~ s/([()\d\-\+\*\/%]+)/$1/gee; # evaluate unroll size 
            
            $_ = unroll($spaces,$codeline,$unrollsize);
        } 
        if ($savelines == 0 and $_ =~ /#GHOST_MUNROLL/) {
            $savelines = 1;
            $savedlines = "";
            $spaces = "";
            next;
        } 
        if ($savelines == 1) {
            if ($_ =~ /#GHOST_MUNROLL#/) { # end of multi-line unrolling
                my $unrollsize = (split /#/,$_)[2];
                $unrollsize =~ s/\\//g; # delete backslash from unrollsize (happens in macros)
                chomp($unrollsize);
                $unrollsize =~ s/([()\d\-\+\*\/%]+)/$1/gee; # evaluate unroll size 
                
                $_ = unroll($spaces,$savedlines,$unrollsize);
                $savelines = 0;
            } else { # save codeline
                $savedlines .= $_;
                next;
            }
        }
        print $_;
    } 
}

sub unroll {
    my $unrolledcode = "";
    for (my $i=0; $i<$_[2]; $i++) {
        my $modcodeline = $_[1];
        $modcodeline =~ s/@/$i/g;
        $modcodeline =~ s/~([()\d\-\+\*\/%]+)~/$1/gee; # evaulate constant expresssion between "~" markers (usually used to construct variable names)
        $unrolledcode = $unrolledcode.$_[0].$modcodeline."\n";
    }
    return $unrolledcode;
}
