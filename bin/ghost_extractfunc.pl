#!/usr/bin/env perl

print "#ifdef __cplusplus\nextern \"C\" {\n#endif\n";

while (<>) {
    if ($_ =~ /(ghost_error_t .+\(.+\))/) {
        chomp($_);
        print "$_;\n";
    }
}

print "#ifdef __cplusplus\n}\n#endif\n";
