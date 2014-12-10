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
            print $funcname."_parameters_t pars;\n";
            print "pars.impl = ".$implementations{$funcpars[0]}.";\n";
            print "pars.mdt = ".$datatypes{$funcpars[1]}.";\n";
            print "pars.vdt = ".$datatypes{$funcpars[2]}.";\n";
            print "pars.storage = ".$storages{$funcpars[3]}.";\n";
            print "pars.chunkheight = ".$funcpars[4].";\n";
            if ($funcpars[5] eq "x") {
                print "pars.blocksz = -1;\n";
            } else {
                print "pars.blocksz = ".$funcpars[5].";\n";
            }
            print $funcname."_kernels[pars] = ".$funcname_full.";\n"; 
            print "}\n";
        } elsif ($funcname eq "ghost_tsmttsm") {
            print "{\n";
            print $funcname."_parameters_t pars;\n";
            print "pars.dt = ".$datatypes{$funcpars[0]}.";\n";
            print "pars.blocksz1 = ".$funcpars[1].";\n";
            if ($funcpars[2] eq "x") {
                print "pars.blocksz2= -1;\n";
            } else {
                print "pars.blocksz2= ".$funcpars[2].";\n";
            }
            print $funcname."_kernels[pars] = ".$funcname_full.";\n"; 
            print "}\n";
        } elsif ($funcname eq "ghost_tsmm") {
            print "{\n";
            print $funcname."_parameters_t pars;\n";
            print "pars.dt = ".$datatypes{$funcpars[0]}.";\n";
            print "pars.blocksz1 = ".$funcpars[1].";\n";
            if ($funcpars[2] eq "x") {
                print "pars.blocksz2= -1;\n";
            } else {
                print "pars.blocksz2= ".$funcpars[2].";\n";
            }
            print $funcname."_kernels[pars] = ".$funcname_full.";\n"; 
            print "}\n";
        } elsif ($funcname eq "ghost_tsmm_inplace") {
            print "{\n";
            print $funcname."_parameters_t pars;\n";
            print "pars.dt = ".$datatypes{$funcpars[0]}.";\n";
            print "pars.blocksz = ".$funcpars[1].";\n";
            print $funcname."_kernels[pars] = ".$funcname_full.";\n"; 
            print "}\n";
        }

    }
}
