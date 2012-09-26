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
move('config.mk','config.mk.orig');
print "succeeded.\n";

my $nopt = $#buildoptions+1; # number of build options
my $maxoptperm = 2**$nopt;   # number of different permutations
my $succeeded = 0;           # number of succeeded permutations

print "  Permuting build options...\n";
for ($optperm=0; $optperm<$maxoptperm; $optperm++)
{
	my @bitmask = split(//,sprintf("%0".$nopt."b", $optperm));

	open (IN,"config.mk.orig") or die "$!";
	open (OUT,">config.mk") or die "$!";

	while ($line = <IN>)
	{
		for ($opt=0; $opt<$nopt; $opt++)
		{
			my $curopt = $buildoptions[$opt];
			$line =~ s/$curopt=[01]/$curopt=$bitmask[$opt]/g;
		}
		print OUT $line;
	}

	close(IN);
	close(OUT);

}
print "  $succeeded/$maxoptperm succeded\n";



chdir("..");

print "=== STAGE 2: CORRECTNESS ===\n"
#remove_tree($libpath,$instpath);

