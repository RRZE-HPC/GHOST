#!/usr/bin/env perl

my %datatypes = (
        'd' => '(ghost_datatype)(GHOST_DT_DOUBLE|GHOST_DT_REAL)',
        's' => '(ghost_datatype)(GHOST_DT_FLOAT|GHOST_DT_REAL)',
        'z' => '(ghost_datatype)(GHOST_DT_DOUBLE|GHOST_DT_COMPLEX)',
        'c' => '(ghost_datatype)(GHOST_DT_FLOAT|GHOST_DT_COMPLEX)',
        'x' => 'GHOST_DT_ANY',
        );
my %storages = (
        'cm' => '(ghost_densemat_storage)(GHOST_DENSEMAT_COLMAJOR)',
        'rm' => '(ghost_densemat_storage)(GHOST_DENSEMAT_ROWMAJOR)',
        );

my %implementations = (
        'plain' => 'GHOST_IMPLEMENTATION_PLAIN',
        'avx' => 'GHOST_IMPLEMENTATION_AVX',
        'avx2' => 'GHOST_IMPLEMENTATION_AVX2',
        'sse' => 'GHOST_IMPLEMENTATION_SSE',
        'mic' => 'GHOST_IMPLEMENTATION_MIC',
        'cuda' => 'GHOST_IMPLEMENTATION_CUDA',
        );

my %alignments = (
        'u' => 'GHOST_UNALIGNED',
        'a' => 'GHOST_ALIGNED'
        );

while (<>) {
    if ($_ =~ /(ghost_error .+\(.+\))/) {
        my @header =  split /[ \(]/,$_;
        my $funcname_full = $header[1];
        my @funcname_parts = split /__/,$funcname_full;
        my $funcname = $funcname_parts[0];
        my @funcnamestart_parts = split /_/,$funcname_full;
        my $funcname_noprefix = $funcname_full;
        $funcname_noprefix =~ s/ghost_//;
        my @funcpars = split /_/,$funcname_parts[1];
        
        if ($funcname eq "ghost_sellspmv") {
            print "{\n";
            print $funcname."_parameters pars;\n";
            print "pars.alignment = ".$alignments{$funcpars[0]}.";\n";
            print "pars.impl = ".$implementations{$funcpars[1]}.";\n";
            print "pars.mdt = ".$datatypes{$funcpars[2]}.";\n";
            print "pars.vdt = ".$datatypes{$funcpars[3]}.";\n";
            print "pars.storage = ".$storages{$funcpars[4]}.";\n";
            print "pars.chunkheight = ".$funcpars[5].";\n";
            if ($funcpars[6] eq "x") {
                print "pars.blocksz = -1;\n";
            } else {
                print "pars.blocksz = ".$funcpars[6].";\n";
            }
            print $funcname."_kernels[pars] = ".$funcname_full.";\n"; 
            if ($funcpars[6] ne "x") {
                print "ghost_gidx nnz;\n";
                print "ghost_gidx nrow;\n";
                print "ghost_sparsemat_nnz(&nnz,mat);\n";
                print "ghost_sparsemat_nrows(&nrow,mat);\n";
                print "ghost_spmv_perf_args spmv_perfargs;\n";
                print "spmv_perfargs.vecncols = rhs->traits.ncols;\n";
                print "spmv_perfargs.globalnnz = nnz;\n";
                print "spmv_perfargs.globalrows = nrow;\n";
                print "spmv_perfargs.dt = rhs->traits.datatype;\n";
                print "spmv_perfargs.flags = traits.flags;\n";
                print "ghost_timing_set_perfFunc(__ghost_functag,\"".$funcname_noprefix."\",ghost_spmv_perf,(void *)&spmv_perfargs,sizeof(spmv_perfargs),GHOST_SPMV_PERF_UNIT);\n";
            }
            print "}\n";
        } elsif ($funcname eq "ghost_dot") {
            print "{\n";
            print $funcname."_parameters pars;\n";
            print "pars.alignment = ".$alignments{$funcpars[0]}.";\n";
            print "pars.impl = ".$implementations{$funcpars[1]}.";\n";
            print "pars.dt = ".$datatypes{$funcpars[2]}.";\n";
            print "pars.storage = ".$storages{$funcpars[3]}.";\n";
            print "pars.blocksz = ".$funcpars[4].";\n";
            print $funcname."_kernels[pars] = ".$funcname_full.";\n";
            if ($funcpars[4] ne "x" ) {
                print "ghost_dot_perf_args dot_perfargs;\n";
                print "dot_perfargs.ncols = ".$funcpars[4].";\n";
                print "if (vec1->context) {\n";
                print "    dot_perfargs.globnrows = vec1->context->gnrows;\n";
                print "} else {\n";
                print "    dot_perfargs.globnrows = vec1->traits.nrows;\n";
                print "}\n";
                print "dot_perfargs.dt = vec1->traits.datatype;\n";
                print "if (vec1 == vec2) {\n";
                print "dot_perfargs.samevec = true;\n";
                print "} else {\n";
                print "dot_perfargs.samevec = false;\n";
                print "}\n";
                print "ghost_timing_set_perfFunc(__ghost_functag,\"".$funcname_noprefix."\",ghost_dot_perf,(void *)&dot_perfargs,sizeof(dot_perfargs),\"GB/s\");\n";
            }
            print "}\n";
        } elsif ($funcname eq "ghost_tsmttsm" or $funcname eq "ghost_tsmttsm_kahan") {
            print "{\n";
            print $funcname."_parameters pars;\n";
            print "pars.alignment = ".$alignments{$funcpars[0]}.";\n";
            print "pars.impl = ".$implementations{$funcpars[1]}.";\n";
            print "pars.dt = ".$datatypes{$funcpars[2]}.";\n";
            if ($funcpars[3] eq "x") {
                print "pars.wcols = -1;\n";
            } else {
                print "pars.wcols = ".$funcpars[3].";\n";
            }
            if ($funcpars[4] eq "x") {
                print "pars.vcols = -1;\n";
            } else {
                print "pars.vcols = ".$funcpars[4].";\n";
            }
            print "pars.unroll = ".$funcpars[5].";\n";
            print "pars.wstor = ".$storages{$funcpars[6]}.";\n";
            print $funcname."_kernels[pars] = ".$funcname_full.";\n"; 
            if ($funcpars[3] ne "x" and $funcpars[4] ne "x" ) {
                print "ghost_gemm_perf_args tsmttsm_perfargs;\n";
                print "tsmttsm_perfargs.n = ".$funcpars[3].";\n";
                print "tsmttsm_perfargs.m = ".$funcpars[4].";\n";
                print "if (v->context) {\n";
                print "    tsmttsm_perfargs.k = v->context->gnrows;\n";
                print "} else {\n";
                print "    tsmttsm_perfargs.k = v->traits.nrows;\n";
                print "}\n";
                print "tsmttsm_perfargs.dt = v->traits.datatype;\n";
                print "if (tsmttsm_perfargs.dt == pars.dt) {\n";
                print "tsmttsm_perfargs.betaiszero = ghost_iszero(beta,pars.dt);\n";
                print "tsmttsm_perfargs.alphaisone = ghost_isone(alpha,pars.dt);\n";
                print "ghost_timing_set_perfFunc(__ghost_functag,\"".$funcname_noprefix."\",ghost_gemm_perf_GBs,(void *)&tsmttsm_perfargs,sizeof(tsmttsm_perfargs),\"GB/s\");\n";
                print "ghost_timing_set_perfFunc(__ghost_functag,\"".$funcname_noprefix."\",ghost_gemm_perf_GFs,(void *)&tsmttsm_perfargs,sizeof(tsmttsm_perfargs),\"GF/s\");\n";
                print "}\n";
            }
            print "}\n";
        } elsif ($funcname eq "ghost_tsmm") {
            print "{\n";
            print $funcname."_parameters pars;\n";
            print "pars.alignment = ".$alignments{$funcpars[0]}.";\n";
            print "pars.impl = ".$implementations{$funcpars[1]}.";\n";
            print "pars.dt = ".$datatypes{$funcpars[2]}.";\n";
            if ($funcpars[3] eq "x") {
                print "pars.xcols = -1;\n";
            } else {
                print "pars.xcols = ".$funcpars[3].";\n";
            }
            if ($funcpars[4] eq "x") {
                print "pars.vcols = -1;\n";
            } else {
                print "pars.vcols = ".$funcpars[4].";\n";
            }
            print "pars.unroll = ".$funcpars[5].";\n";
            print "pars.multipleof = ".$funcpars[6].";\n";
            print "pars.xstor = ".$storages{$funcpars[7]}.";\n";
            print $funcname."_kernels[pars] = ".$funcname_full.";\n";
            if ($funcpars[3] ne "x" and $funcpars[4] ne "x" ) {
                print "ghost_gemm_perf_args tsmm_perfargs;\n";
                print "tsmm_perfargs.n = ".$funcpars[3].";\n";
                print "tsmm_perfargs.k = ".$funcpars[4].";\n";
                print "if (v->context) {\n";
                print "    tsmm_perfargs.m = v->context->gnrows;\n";
                print "} else {\n";
                print "    tsmm_perfargs.m = v->traits.nrows;\n";
                print "}\n";
                print "tsmm_perfargs.dt = x->traits.datatype;\n";
                print "tsmm_perfargs.betaiszero = ghost_iszero(beta,x->traits.datatype);\n";
                print "tsmm_perfargs.alphaisone = ghost_isone(alpha,x->traits.datatype);\n";
                print "ghost_timing_set_perfFunc(__ghost_functag,\"".$funcname_noprefix."\",ghost_gemm_perf_GBs,(void *)&tsmm_perfargs,sizeof(tsmm_perfargs),\"GB/s\");\n";
                print "ghost_timing_set_perfFunc(__ghost_functag,\"".$funcname_noprefix."\",ghost_gemm_perf_GFs,(void *)&tsmm_perfargs,sizeof(tsmm_perfargs),\"GF/s\");\n";
            }
            print "}\n";
        } elsif ($funcname eq "ghost_tsmm_inplace") {
            print "{\n";
            print $funcname."_parameters pars;\n";
            print "pars.alignment = ".$alignments{$funcpars[0]}.";\n";
            print "pars.impl = ".$implementations{$funcpars[1]}.";\n";
            print "pars.dt = ".$datatypes{$funcpars[2]}.";\n";
            if ($funcpars[3] eq "x") {
                print "pars.ncolsin = -1;\n";
            } else {
                print "pars.ncolsin = ".$funcpars[3].";\n";
            }
            if ($funcpars[4] eq "x") {
                print "pars.ncolsout = -1;\n";
            } else {
                print "pars.ncolsout = ".$funcpars[4].";\n";
            }
            print $funcname."_kernels[pars] = ".$funcname_full.";\n"; 
            if ($funcpars[3] ne "x" and $funcpars[4] ne "x" ) {
                print "ghost_gemm_perf_args tsmm_perfargs;\n";
                print "tsmm_perfargs.n = ".$funcpars[3].";\n";
                print "tsmm_perfargs.k = ".$funcpars[4].";\n";
                print "if (x->context) {\n";
                print "    tsmm_perfargs.m = x->context->gnrows;\n";
                print "} else {\n";
                print "    tsmm_perfargs.m = x->traits.nrows;\n";
                print "}\n";
                print "tsmm_perfargs.dt = x->traits.datatype;\n";
                print "tsmm_perfargs.betaiszero = ghost_iszero(beta,x->traits.datatype);\n";
                print "tsmm_perfargs.alphaisone = ghost_isone(alpha,x->traits.datatype);\n";
                print "tsmm_perfargs.aisc = true;\n";
                print "ghost_timing_set_perfFunc(__ghost_functag,\"".$funcname_noprefix."\",ghost_gemm_perf_GBs,(void *)&tsmm_perfargs,sizeof(tsmm_perfargs),\"GB/s\");\n";
                print "ghost_timing_set_perfFunc(__ghost_functag,\"".$funcname_noprefix."\",ghost_gemm_perf_GFs,(void *)&tsmm_perfargs,sizeof(tsmm_perfargs),\"GF/s\");\n";
            }
            print "}\n";
        }

    }
}
