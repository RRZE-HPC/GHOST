#!/usr/bin/perl
use File::Copy::Recursive qw(dircopy);
use File::Path qw(remove_tree);
use Cwd;

my $srcpath="/home/hpc/unrz/unrza317/proj/SpMVM/libspmvm";
my $libpath=getcwd()."/libspmvm";
my $instpath=getcwd()."/inst";
my @buildoptions=("DOUBLE","COMPLEX","OPENCL");


print "Copying the source code from $srcpath to $libpath... ";
dircopy($srcpath,$libpath) or die "$!";
print "done.\n";


chdir("./libspmvm");

my $nopt = $#buildoptions+1;
my $maxoptperm = 2**$nopt;

for ($optperm=0; $optperm<$maxoptperm; $optperm++)
{
	my @bitmask = split(//,sprintf("%0".$nopt."b", $optperm));

	print "Building with ";
	for ($opt=0; $opt<$nopt; $opt++)
	{
		print $buildoptions[$opt]."=".$bitmask[$opt]." ";

	}
	print "\n";
}



chdir("..");
remove_tree($libpath,$instpath);

