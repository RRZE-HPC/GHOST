use strict;
use warnings;

my $function;
my @variants;

while (<>) {
    if (/#GHOST_FUNC_BEGIN/../#GHOST_FUNC_END/) {
        if (/#GHOST_FUNC_BEGIN/) {
            $function = "";
            @variants = split /[,#]/,$_; 
            next;
        } elsif (not /#GHOST_FUNC_END/) {
            $function = $function.$_;
        } else {
            for (@variants[2..$#variants]) {
                chomp($_);
                my $f = $function;
                $f =~ s/\$\/([\d]*)/$_ \/ $1/ge; # division
                $f =~ s/\$/$_/g;
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
    
    $_[0] = "";
    for (my $i=0; $i<$unrollsize; $i++) {
        my $modcodeline = $codeline;
        $modcodeline =~ s/@/$i/g;
        $_[0] = $_[0].$spaces.$modcodeline."\n";
    }
}
