#!/usr/bin/env perl

while (<>) {
    if ($_ =~ /(ghost_error_t .+\(.+\))/) {
        chomp($_);
        print "$_;\n";
    }
}
