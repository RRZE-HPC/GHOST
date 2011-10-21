#!/bin/tcsh

module load intel64/11.1
module load intelmpi

#set pinlist = ( 0,1,2,3,4,5,6,7,8,9,10,11 0,1,2,3,4,5_6,7,8,9,10,11 0,1,2,3,4,5,6,7,8,9,10,11,15 0,1,2,3,4,5,15_6,7,8,9,10,11,20 0_1_2_3_4_5_6_7_8_9_10_11 0,12_1,13_2,14_3,15_4,16_5,17_6,18_7,19_8,20_9,21_10,22_11,23 )
set pinlist = ( 0,1 ) 
 
set distscheme = ( NZE LNZ ROW)
set mymasks    = ( 262143 )
#set mymasks    = ( 99945 99945 98304 99945 99945 98304 )
set mults      = "50"
set work_dist  = "1"
set thisexe    = "./HybridSpMVM_TUNE_SETUP_REVBUF_NLDD_PLACE_CYCLES_INT_woody.x"
set thismpi    = "/apps/rrze/bin/mpirun_rrze-intelmpd"
#set thispin    = "/apps/likwid/likwid-2.1/bin/likwid-pin"
set thispin    = "/apps/likwid/devel/bin/likwid-pin"
set master     = `head -n 1 ${PBS_NODEFILE}`
set skip       = "-s 0x3"
set io_format  = "2"
set outerit    = "1"
set datadir    = "./test"
set wd_flag    = $distscheme[$work_dist]

echo "rrze-mpi:"
$thismpi --version
echo "mpi:"
mpirun --version

mkdir $datadir
setenv KMP_AFFINITY disabled 
setenv OMP_SCHEDULE static
#setenv LD_PRELOAD /apps/rrze/lib/ptoverride-ubuntu64.so

#echo $LD_PRELOAD

foreach mat ( dlr2 ) 

foreach nodes ( 1 )

   foreach jobtype ( 1 )
         
      set jobmask      = $mymasks[$jobtype]

      set realmults    = `echo ${nodes} | awk '{print $0*'${mults}'}'` 
      set fixed        = "${thisexe} ${realmults} 1 ${mat} ${io_format} ${work_dist} ${jobmask} ${outerit}"
      set threadsperPE = `echo $pinlist[$jobtype] | sed -e 's/_/ /' | awk '{print $1}' | sed -e 's/,/ /g' | awk '{print NF}'`
      set lastthread   = `echo $threadsperPE | awk '{print $0-1}'`
      set PEpernode    = `echo $pinlist[$jobtype] | sed -e 's/_/ /g' | awk '{print NF}'`
      set np           = `echo $nodes | awk '{print $0*'${PEpernode}'}'`
      set root         = "${master}_${mat}_NZE_${nodes}x${PEpernode}x${threadsperPE}.out"
      set ivargs       = "-npernode ${PEpernode} -np ${np}"

      if ( $threadsperPE == 1 ) then 
         set pin2cores = "0"
      else
         set pin2cores = "L:0-${lastthread}"
      endif

      set pinlikwid     = "${thispin} ${skip} -c ${pin2cores}"

      set command       = "time ${thismpi} -intelmpd -pin $pinlist[$jobtype] ${ivargs} ${pinlikwid} ${fixed} "

      echo "#nodes:" ${nodes} "pinmask:" $pinlist[$jobtype]
      echo $command

      setenv OMP_NUM_THREADS ${threadsperPE}

      $command > ${datadir}/Performance_${root}

   end
end

end
