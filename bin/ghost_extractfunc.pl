#!/usr/bin/env perl

print "#ifdef __cplusplus\nextern \"C\" {\n#endif\n";

if (@ARGV) {
    while (<>) {
        if ($_ =~ /(ghost_error .+\(.+\))/ and $_ !~ /static/) {
            $_ =~ s/[{]+//g;
            chomp($_);
            print "$_;\n";
        }
    }
}

print "#ifdef __cplusplus\n}\n#endif\n";
