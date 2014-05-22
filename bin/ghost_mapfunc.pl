#!/usr/bin/env perl

my %datatypes = (
        'd' => '(ghost_datatype_t)(GHOST_DT_DOUBLE|GHOST_DT_REAL)',
        's' => '(ghost_datatype_t)(GHOST_DT_FLOAT|GHOST_DT_REAL)',
        'z' => '(ghost_datatype_t)(GHOST_DT_DOUBLE|GHOST_DT_COMPLEX)',
        'c' => '(ghost_datatype_t)(GHOST_DT_FLOAT|GHOST_DT_COMPLEX)',
        );
my %storages = (
        'cm' => '(ghost_densemat_storage_t)(GHOST_DENSEMAT_COLMAJOR)',
        'rm' => '(ghost_densemat_storage_t)(GHOST_DENSEMAT_ROWMAJOR)',
        );

my %implementations = (
        'avx' => 'GHOST_IMPLEMENTATION_AVX',
        'sse' => 'GHOST_IMPLEMENTATION_SSE',
        'mic' => 'GHOST_IMPLEMENTATION_MIC',
        );

while (<>) {
    if ($_ =~ /(ghost_error_t .+\(.+\))/) {
        my @header =  split /[ \(]/,$_;
        my $funcname_full = $header[1];
        my @funcname_parts = split /__/,$funcname_full;
        my $funcname = $funcname_parts[0];
        my @funcpars = split /_/,$funcname_parts[1];
        
        if ($funcname eq "ghost_sellspmv") {
            print "{\n";
            print $funcname."_parameters_t pars = {";
            print ".impl = ".$implementations{$funcpars[0]}.", ";
            print ".mdt = ".$datatypes{$funcpars[1]}.", ";
            print ".vdt = ".$datatypes{$funcpars[2]}.", ";
            print ".storage = ".$storages{$funcpars[3]}.", ";
            print ".chunkheight = ".$funcpars[4].", ";
            if ($funcpars[5] eq "x") {
                print ".blocksz = -1";
            } else {
                print ".blocksz = ".$funcpars[5];
            }
            print "};\n";
            print $funcname."_kernels[pars] = ".$funcname_full.";\n"; 
            print "}\n";
        } elsif ($funcname eq "ghost_tsmm" or $funcname eq "ghost_tsmttsm") {
            print "{\n";
            print $funcname."_parameters_t pars = {";
            print ".dt = ".$datatypes{$funcpars[0]}.", ";
            print ".blocksz = ".$funcpars[1];
            print "};\n";
            print $funcname."_kernels[pars] = ".$funcname_full.";\n"; 
            print "}\n";
        }

    }
}
