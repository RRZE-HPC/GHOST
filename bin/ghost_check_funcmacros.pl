#!/usr/bin/env perl

my $afterhead = 0;
my $infunc = 0;
my $funchead;
my $hasenter = 0;
my $hasexit = 0;

while (<>) {
    if (/^(ghost_error|void|int|bool|char \*) ghost_(\w+)\s*\([^)]*\)\s*/) {
        $funchead = $_;
        $afterhead = 1;
        $hasenter = 0;
        $hasexit = 0;
        next;
    }
    if ($afterhead and /^{/) {
        $afterhead = 0;
        $infunc = 1;
        next;
    }
    if ($infunc and /^}/) {
        $infunc = 0;
        if (not $hasenter or not $hasexit) {
            print "Missing function macro(s): $funchead";
        }
    }
    if ($infunc) {
        if (/\s*GHOST_FUNC_ENTER/) {
            $hasenter = 1;
        }
        if (/\s*GHOST_FUNC_EXIT/) {
            $hasexit = 1;
        }
    }
}
        

