#!/usr/bin/perl
use File::Copy::Recursive qw(dircopy);
use File::Copy;
use File::Path qw(remove_tree);
use Cwd qw(realpath getcwd);

my @buildoptions=("DOUBLE","COMPLEX");

####### DO NOT EDIT BELOW

my $srcpath=realpath('..')."/libspmvm";
my $libpath=getcwd()."/libspmvm";
my $instpath=getcwd()."/inst";

#print "=== STAGE 1: BUILD ===\n";

#print "  Preparation... ";
dircopy($srcpath,$libpath) or die "$!";
chdir("./libspmvm");
copy('config.mk','config.mk.orig');
#print "succeeded.\n";

my $nopt = $#buildoptions+1; # number of build options
my $maxoptperm = 2**$nopt;   # number of different permutations
my $succeeded = 0;           # number of succeeded permutations

#print "  Testing configurations... ";
unlink('../make.log');
unlink('../run.log');
for ($optperm=$maxoptperm-1; $optperm<$maxoptperm; $optperm++)
{
	my @bitmask = split(//,sprintf("%0".$nopt."b", $optperm));

	copy('config.mk.orig','config.mk');   # restore original config.mk
	open (OUT,">>config.mk") or die "$!"; # append new options to file (overrides)

	for ($opt=0; $opt<$nopt; $opt++)
	{
		print OUT $buildoptions[$opt]."=".$bitmask[$opt]."\n";
		print $buildoptions[$opt]."=".$bitmask[$opt]." ";
	}
	print "\n";
	print OUT "INSTDIR=".$instpath."\n";
	close(OUT);

	system('make distclean uninstall > /dev/null');
	if (system('make install >> ../make.log 2>&1') == 0) {
		print "  -> build: OK\n";
		$succeeded++;
	} else {
		print "  -> build: FAILURE\n";
		next;
	}
		

	my $numKernels;
	my $numOptions;

	open (IN,$instpath."/include/spmvm.h");
	while (<IN>) {
		if ($_ =~ /SPMVM_NUMKERNELS\s+([0-9]+)/) {
			$numKernels = $1;
		}
		if ($_ =~ /SPMVM_NUMOPTIONS\s+([0-9]+)/) {
			$numOptions = $1;
		}
	}
	$ntest += $numKernels;

	
	chdir("..");
	system('make OPENCL=0 > /dev/null');

#TODO number of actually(!) executed kernels

	my $totKern = 2**$numOptions*$numKernels;
	my $sucKern = 0;

	for ($runopt=0; $runopt<2**$numOptions; $runopt++) {
		my $nErr = system('./test.x /home/vault/unrz/unrza317/matrices/test1/test1_double_CRS_bin.dat '.$runopt.' >> ./run.log 2>&1')/256;
		$sucKern+=$numKernels-$nErr;
	}

	print "  -> run  : ";
	print $sucKern==$totKern?"OK\n":"$sucKern/$totKern\n";

	system('make distclean > /dev/null');
	
	
	chdir("./libspmvm");

}



chdir("..");

#print "=== STAGE 2: CORRECTNESS ===\n";
#remove_tree($libpath,$instpath);

