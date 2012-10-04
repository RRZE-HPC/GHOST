#!/usr/bin/perl
use PBS::Client;
use Chart::Graph::Xmgrace qw(xmgrace);

my $NNODES = @ARGV[0];
my $CMD = "module load intel64\n\n".
"for i in {1..$NNODES}\n".
"do\n".
"\tmpiexec.hydra -n \$i -ppn 1 apps/spmvm/spmvm.x /lxfs/unrz/unrza317/rrze3_vv_d.crs\n".
"done\n";

my $client = PBS::Client->new();
my $job = PBS::Client::Job->new(
		nodes=>$NNODES,
		ppn=>24,
		shell=>'/bin/bash --login',
		wallt=>'00:10:00',
		cmd=>$CMD);
$client->qsub($job);
my $jid = $job->pbsid;
print "Starting job ".$jid." with ".$NNODES." nodes... ";
while (`qstat $jid 2>/dev/null | wc -l` > 0) {
	sleep(1); # wait until job is finished
}
print "finished\n";

#my $jid = 462223;
open (IN,"<pbsjob.sh.o".$jid);
my @nodes;
my @serial;
my @vector;
my @gfhybr;
my @taskmd;

while (<IN>) {
	my $line = $_;
	if ($line =~ /Nodes\s*:\s*([0-9]+)/) {
		push(@nodes,$1);
		print "$1 ";
	}
	if ($line =~ /non-MPI:/) {
		if ($line =~ /non-MPI: SUCCESS @\s+([0-9]+\.[0-9]+)\s*GF\/s/) {
			push(@serial,$1);
		} else {
			push(@serial," ");
		}
	}
	if ($line =~ /vector mode:/) {
		if ($line =~ /vector mode: SUCCESS @\s+([0-9]+\.[0-9]+)\s*GF\/s/) {
			push(@vector,$1);
		} else {
			push(@vector," ");
		}
	}
	if ($line =~ /g\/f hybrid:/) {
		if ($line =~ /g\/f hybrid: SUCCESS @\s+([0-9]+\.[0-9]+)\s*GF\/s/) {
			push(@gfhybr,$1);
		} else {
			push(@gfhybr," ");
		}
	}
	if ($line =~ /task mode/) {
		if ($line =~ /task mode: SUCCESS @\s+([0-9]+\.[0-9]+)\s*GF\/s/) {
			push(@taskmd,$1);
		}else {
			push(@taskmd," ");
		}
	}
}

my @serialPerf;
my @vectorPerf;
my @gfhybrPerf;
my @taskmdPerf;

foreach $n (@nodes) {

	my @entry = ($n,$serial[0]);
	push(@serialPerf,\@entry);
	my @entry = ($n,$vector[0]);
	push(@vectorPerf,\@entry);
	my @entry = ($n,$gfhybr[0]);
	push(@gfhybrPerf,\@entry);
	my @entry = ($n,$taskmd[0]);
	push(@taskmdPerf,\@entry);

	shift(@serial);
	shift(@vector);
	shift(@gfhybr);
	shift(@taskmd);

}

xmgrace( { "title" => "Example of a XY Chart",
		"subtitle" =>"optional subtitle",
		"type of graph" => "XY chart",
		"output type" => "png",
		"output file" => "xmgrace1.png",
		"x-axis label" => "Nodes",
		"y-axis label" => "GF/s",
		"grace output file" => "xmgrace1.agr",
		},

		[{"data format" => "matrix",
		"title"=>"Vector mode"},
		\@vectorPerf
		],
		[{"data format" => "matrix",
		"title"=>"Good faith hybrid"},
		\@gfhybrPerf
		],
		[{"data format" => "matrix",
		"title"=>"Task mode"},
		\@taskmdPerf
]

);

`display xmgrace1.png &`;


