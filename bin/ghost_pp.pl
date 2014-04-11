#!/usr/bin/env perl

use strict;
use warnings;
use List::Permutor;

my $function;
my %variants;

#my %list = (
#        "a"=>(1,2),
#        "b"=>(3,4),
#        "c"=>(5,6),
#        "d"=>(7,8));
#permute(\%list);

sub create_permutations {
    my $npar = $_[0];
    my $permutor = List::Permutor->new(@{$_[1]});
    my @vars;
    while (my @permutation = $permutor->peek()) {
        my @firsttags = @permutation[0..$npar-1];
        for (@firsttags) {
            $_ =~ s/=[\d]+//g;
        }

        my @uniquefirsttags = uniq(@firsttags);
        my $nuniquefirsttags = $#uniquefirsttags+1;
        if ($nuniquefirsttags == $npar) {
            my @firsttags = @permutation[0..$npar-1];
            @firsttags = sort @firsttags;
            push(@vars, join(',',@firsttags))
        }
        @permutation = $permutor->next();
    }

    @vars = uniq(@vars);
    for (@vars) {
#        print $_."\n";
    }

    return @vars;
}

sub uniq {
    return keys %{{map{$_ => 1} @_ }};
}

while (<>) {
    if (/#GHOST_FUNC_BEGIN/../#GHOST_FUNC_END/) {
        if (/#GHOST_FUNC_BEGIN/) {
            $function = "";
            undef %variants;
            my @parameters = split /#/,$_;
            for (@parameters[2..$#parameters]) {
                my @parinfo = split /=/,$_;
                my @varlist = split /,/,$parinfo[1];
#                print $parinfo[0]."=";
                foreach my $var (@varlist) {
                    chomp($var);
                    push(@{$variants{$parinfo[0]}},$var);
#                    print $var." ";
                }
#                print "\n";
            }
# @variants = split /[,#]/,$_; 
            next;
        } elsif (not /#GHOST_FUNC_END/) {
            $function = $function.$_;
        } else {
            my @parlist;
            foreach my $tag (keys %variants) {
                foreach my $var (@{$variants{$tag}}) {
# print ">>> ".$tag." -> ".$var."\n";
                    push (@parlist,$tag."=".$var);
                }
            }
# my $permutor = List::Permutor->new(

#            print "call w/ @parlist\n";
            my @actualvariants = create_permutations(scalar keys %variants,\@parlist);
            
            foreach my $actualvariant (@actualvariants) {
                my $f = $function;
                my @varlist = split /,/,$actualvariant;
                foreach my $actualparameter (@varlist) {
                    my $tag = (split /=/,$actualparameter)[0];
                    my $subst = (split /=/,$actualparameter)[1];
#                    print ">>> ".$tag." --- ".$subst."\n";
                    $f =~ s/$tag/$subst/g;
                }
                $f =~ s/([\d]+)\*([\d]+)\/([\d]+)/$1 * $2 \/ $3/ge; # evaulate a*b/c
                $f =~ s/([\d]+)\/([\d]+)/$1 \/ $2/ge; # evaulate a/b
                for (split /^/,$f) {
                    if ($_ =~ /#GHOST_UNROLL/) {
                        unroll($_);
                    } 
                    print $_;
                }
#                $f =~ s/\$\/([\d]+)/$_ \/ $1/ge; # div
#                $f =~ s/([\d]+)\*\$/$1 * $_/ge; # mul
#                $f =~ s/\$\*([\d]+)/$_ * $1/ge; # mul
#                $f =~ s/([\d]+)\/([\d]+)/$1 \/ $2/ge; # div
#                $f =~ s/([\d]+)\*([\d]+)/$1 * $2/ge; # mul
#                    $f =~ s/CHUNKHEIGHT/$ch/g;
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

#sub permute {
#    my @list = sort keys %{$_[0]};
#    my $len = $#list+1;
#print "len ".$len."\n";
#    if ($len == 1) {
#        print "LENGTH ONE $list[0]\n";
#        print $list[0]." ";
#    } else {
#        print "$list[0] ";
#        delete $list{(sort keys %{$_[0]})[0]};
#        permute(\%list);
#    }
#}

sub unroll {
    my $spaces =  (split /#/,$_[0])[0];
    my $codeline =  (split /#/,$_[0])[2];
    my $unrollsize = (split /#/,$_[0])[3];
    chomp($unrollsize);
    
    $_[0] = "";
    for (my $i=0; $i<$unrollsize; $i++) {
        my $modcodeline = $codeline;
        $modcodeline =~ s/@/$i/g;
        $_[0] = $_[0].$spaces.$modcodeline."\n";
    }
}
