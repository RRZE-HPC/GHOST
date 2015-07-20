#!/usr/bin/env perl

use strict;
use warnings;
use integer;

my $function;
my %vrs; # all variants, array of hashes
my @prs; # sorted parameters
my @idx; # index into vrs
my @assignments;


sub create_assignments {
    if ($idx[$_[0]] == $#{$vrs{$prs[$_[0]]}}) { # this value is max: set to 0 and go to next value
        if ($_[0] < $#prs) { # there is a next value
            create_assignments($_[0]+1);
        } else { #iteration is finished
            my @mapping;
            for (my $i=0; $i < scalar @prs; $i++) {
                push (@mapping,$prs[$i]);
                push (@mapping,${$vrs{$prs[$i]}}[$idx[$i]]);
            }
            push (@assignments,\@mapping);
            return;
        }
    } else { # increment this value
        my @mapping;
        for (my $i=0; $i < scalar @prs; $i++) {
            push (@mapping,$prs[$i]);
            push (@mapping,${$vrs{$prs[$i]}}[$idx[$i]]);
        }
        push (@assignments,\@mapping);
        $idx[$_[0]]++;
        for (my $i=0; $i < $_[0]; $i++) {
            $idx[$i] = 0;
        }
        
        create_assignments(0);
    }
}
    
while (<>) {
    if (/#GHOST_FUNC_BEGIN/../#GHOST_FUNC_END/) {
        if (/#GHOST_FUNC_BEGIN/) {
            $function = "";
            undef %vrs;
            undef @assignments;

            my @parameters = split /#/,$_;
            for (@parameters[2..$#parameters]) {
                my @parinfo = split /=/,$_;
                my @varlist = split /,/,$parinfo[1];
                foreach my $var (@varlist) {
                    chomp($var);
                    push(@{$vrs{$parinfo[0]}},$var);
                }
                $idx[(scalar keys %vrs)-1] = 0;
            }
            @prs = sort keys %vrs;
            create_assignments(0);
            next;
        } elsif (not /#GHOST_FUNC_END/) {
            $function = $function.$_;
        } else {
            foreach my $assignment (@assignments) {
                my $f = $function;
                for (my $i = 0; $i < (scalar @$assignment)/2; $i++) {
                    $f =~ s/@$assignment[2*$i]/@$assignment[2*$i+1]/g;
                }
                $f =~ s/([\d]+)\*([\d]+)\/([\d]+)/int($1 * $2 \/ $3)/ge; # evaulate a*b/c
                $f =~ s/\(([\d]+)\+([\d]+)\)\/([\d]+)/int(($1 + $2) \/ $3)/ge; # evaulate (a+b)/c
                $f =~ s/([\d]+)\/([\d]+)/int($1 \/ $2)/ge; # evaulate a/b
                $f =~ s/([\d]+)%([\d]+)/int($1 % $2)/ge; # evaulate a%b
                $f =~ s/([\d]+)-([\d]+)/$1 - $2/ge; # evaulate a-b
                for (split /^/,$f) {
                    if ($_ =~ /#GHOST_UNROLL/) {
                        unroll($_);
                    } 
                    print $_;
                }
                print("\n");
            } 
            next;
        }
    } else {
        if (/#GHOST_UNROLL/) {
            unroll($_);
        }
        print $_;
    } 
}

sub unroll {
    my $spaces =  (split /#/,$_[0])[0];
    my $codeline =  (split /#/,$_[0])[2];
    my $unrollsize = (split /#/,$_[0])[3];
    chomp($unrollsize);
    $unrollsize =~ s/([\d]+)\*([\d]+)/$1 * $2/ge; # evaulate a*b
    
    $_[0] = "";
    for (my $i=0; $i<$unrollsize; $i++) {
        my $modcodeline = $codeline;
        $modcodeline =~ s/@/$i/g;
        $modcodeline =~ s/([\d]+)\*([\d]+)\+([\d]+)/$1 * $2 + $3/ge; # evaulate a*b+c
        $modcodeline =~ s/([\d]+)\*([\d]+)/$1 * $2/ge; # evaulate a*b
        $modcodeline =~ s/([\d]+)\+([\d]+)/$1 + $2/ge; # evaulate a+b
        $modcodeline =~ s/([\d]+)\/([\d]+)/$1 \/ $2/ge; # evaulate a/b
        $modcodeline =~ s/([\d]+)%([\d]+)/$1 % $2/ge; # evaulate a%b
        $_[0] = $_[0].$spaces.$modcodeline."\n";
    }
}
