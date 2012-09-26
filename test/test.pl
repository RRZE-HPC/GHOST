#!/usr/bin/perl
use File::Copy::Recursive qw(dircopy);
use File::Copy;
use File::Path qw(remove_tree);
use Cwd qw(realpath getcwd);

my @buildoptions=("DOUBLE","COMPLEX","OPENCL");

####### DO NOT EDIT BELOW

my $srcpath=realpath('..')."/libspmvm";
my $libpath=getcwd()."/libspmvm";
my $instpath=getcwd()."/inst";

print "=== STAGE 1: BUILD ===\n";

print "  Preparation... ";
dircopy($srcpath,$libpath) or die "$!";
chdir("./libspmvm");
copy('config.mk','config.mk.orig');
print "succeeded.\n";

my $nopt = $#buildoptions+1; # number of build options
my $maxoptperm = 2**$nopt;   # number of different permutations
my $succeeded = 0;           # number of succeeded permutations

print "  Permuting build options...\n";
for ($optperm=0; $optperm<$maxoptperm; $optperm++)
{
	my @bitmask = split(//,sprintf("%0".$nopt."b", $optperm));

	copy('config.mk.orig','config.mk');   # restore original config.mk
	open (OUT,">>config.mk") or die "$!"; # append new options to file (overrides)

	for ($opt=0; $opt<$nopt; $opt++)
	{
		print OUT $buildoptions[$opt]."=".$bitmask[$opt]."\n";
	}

	close(OUT);

# make
		$succeeded++;

}
print "  $succeeded/$maxoptperm succeded\n";



chdir("..");

print "=== STAGE 2: CORRECTNESS ===\n";
remove_tree($libpath,$instpath);

