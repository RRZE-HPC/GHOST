#!/bin/tcsh

#PBS -lnodes=4:ppn=16,walltime=01:00:00

cd /home/woody/unrz/unrza337/mucosim/SpMVM/SpMVM/Hybrid

module load intel64/11.1
module load cuda

#set pinlist = ( 0,1,2,3,4,5,6,7,8,9,10,11 0,1,2,3,4,5_6,7,8,9,10,11 0,1,2,3,4,5,6,7,8,9,10,11,15 0,1,2,3,4,5,15_6,7,8,9,10,11,20 0_1_2_3_4_5_6_7_8_9_10_11 0,12_1,13_2,14_3,15_4,16_5,17_6,18_7,19_8,20_9,21_10,22_11,23 )
set pinlist = ( 0,1 0,1 0,1 ) 
#set pinlist = ( 0,1_4,5 0,1_4,5 0,1_4,5 ) 
 
set distscheme = ( NZE LNZ ROW ) # MPI distribution of matrix: equal_NZE equal_NZE_opt equal_row
set mymasks    = ( 502 261640 262142 ) # kernel selector: full_only split_only all
set mults      = "20"
set work_dist  = "1" # 1 approx eq NZE / 2 fancy NZE / else eq NRows
set thisexe    = "./HybridSpMVM_TUNE_SETUP_REVBUF_NLDD_CUDAKERNEL_PLACE_CYCLES_INT_gpu.x"
set thismpi    = "/apps/rrze/bin/mpirun_rrze-intelmpd" 
#set thismpi    = "mpirun"
set thispin    = "/apps/likwid/devel/bin/likwid-pin"
set master     = `head -n 1 ${PBS_NODEFILE}`
set skip       = "-s 0x3"
set io_format  = "2" # 1 MPI-parallel / 2 binary / else ASCII
set outerit    = "1"
set datadir    = "./test"
set wd_flag    = $distscheme[$work_dist]

mkdir $datadir
setenv KMP_AFFINITY disabled 
setenv OMP_SCHEDULE static
foreach mat ( rrze3 ) # rrze3_vv rrze3_rcm dlr1 dlr2 dlr3 ) 

foreach nodes ( 2 )

   foreach jobtype ( 3 )
    foreach gridDim ( 512 ) # 256 512 1024 )
      foreach blockDim( 512 )
      set jobmask      = $mymasks[$jobtype]

      set realmults    = `echo ${nodes} | awk '{print $0*'${mults}'}'` 
      set fixed        = "${thisexe} ${realmults} 1 ${mat} ${io_format} ${work_dist} ${jobmask} ${outerit}"
      set kernelparam  = "${gridDim} ${blockDim}"
      set threadsperPE = `echo $pinlist[$jobtype] | sed -e 's/_/ /' | awk '{print $1}' | sed -e 's/,/ /g' | awk '{print NF}'`
      set lastthread   = `echo $threadsperPE | awk '{print $0-1}'`
      set PEpernode    = `echo $pinlist[$jobtype] | sed -e 's/_/ /g' | awk '{print NF}'`
      set np           = `echo $nodes | awk '{print $0*'${PEpernode}'}'`
      set root         = "${master}_${mat}_NZE_${nodes}x${PEpernode}x${threadsperPE}_${gridDim}x${blockDim}.out"
      set ivargs       = "-npernode ${PEpernode} -n ${np}"
      
      if ( $threadsperPE == 1 ) then 
         set pin2cores = "L:0"
      else
      set pin2cores = "L:0-${lastthread}" # `echo $pinlist[$jobtype] | sed -e 's/_/,/g'`
      endif
   
      set pinlikwid    = "${thispin} ${skip} -c ${pin2cores}"
      #set pinlikwid     = "env LD_PRELOAD=/apps/rrze/lib/ptoverride-ubuntu64.so" # "${thispin} ${skip} -c ${pin2cores} -t intel"

      set command       = "time ${thismpi} -intelmpd -pin $pinlist[$jobtype] ${ivargs} ${pinlikwid} ${fixed} ${kernelparam}"

      echo "#nodes:" ${nodes} "pinmask:" $pinlist[$jobtype]
      mpirun --version
      echo $command

      setenv OMP_NUM_THREADS ${threadsperPE}
      setenv I_MPI_DEVICE rdssm 

      $command > ${datadir}/Performance_${root}
      #echo "${gridDim} ${blockDim}" >> perfmeasure.dat
      #grep "Kern No. " ${datadir}/Performance_${root} >> perfmeasure.dat
    end
    end
   end
end

end
