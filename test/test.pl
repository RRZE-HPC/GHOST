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

my $nopt = $#buildoptions+1; # number of build options
my $maxoptperm = 2**$nopt;   # number of different permutations

#print "  Testing configurations... ";
unlink('../make.stderr');
unlink('../make.stdout');
unlink('../run.stderr');
unlink('../run.stdout');
for ($optperm=0; $optperm<$maxoptperm; $optperm++)
{
	my @bitmask = split(//,sprintf("%0".$nopt."b", $optperm));

	chdir("./libspmvm");
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
	if (system('make install 2>> ../make.stderr 1>> ../make.stdout') == 0) {
		print "  -> build: OK\n";
	} else {
		print "  -> build: FAILURE\n";
		next;
	}
		

	chdir("..");
	system('make OPENCL=0 > /dev/null');

	my @out = `./test.x /home/vault/unrz/unrza317/matrices/test1/test1_double_CRS_bin.dat 2>run.stderr`;
	open (OUT,">run.stdout") or die "$!";

	my $suc = 0;
	my $err = 0;
	for my $line (@out) {
		if ($line =~ m/^0/) {
			$err++;
		} elsif ($line =~ m/^1/) {
			$suc++;
		}
		print OUT $line;
	}
	close(OUT);

	print "  -> run  : ";
	print $err==0?"OK\n":"$suc/".$suc+$err."\n";

	system('make distclean > /dev/null');
	
}



chdir("..");

#print "=== STAGE 2: CORRECTNESS ===\n";
remove_tree($libpath,$instpath);

