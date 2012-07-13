#!/bin/tcsh

#PBS -lnodes=8:ppn=16,walltime=01:00:00
#PBS -N my_job
#PBS -e my_job.$PBS_JOBID.err
#PBS -o my_job.$PBS_JOBID.out
#PBS -V


cd /global/homes/w/wellein/MK/Hybrid-stripped

module load openmpi-intel
module load intel
module load cuda

#set pinlist = ( 0,1,2,3,4,5,6,7,8,9,10,11 0,1,2,3,4,5_6,7,8,9,10,11 0,1,2,3,4,5,6,7,8,9,10,11,15 0,1,2,3,4,5,15_6,7,8,9,10,11,20 0_1_2_3_4_5_6_7_8_9_10_11 0,12_1,13_2,14_3,15_4,16_5,17_6,18_7,19_8,20_9,21_10,22_11,23 )
set pinlist = ( 0,1 0,1 0,1 0,1 0,1 0,1 ) # 1 gpu pro knoten
#set pinlist = ( 0,1_4,5 0,1_4,5 0,1_4,5 0,1_4,5 0,1_4,5 0,1_4,5 ) # 2 gpus pro knoten
 
set distscheme = ( NZE LNZ ROW ) # MPI distribution of matrix: equal_NZE equal_NZE_opt equal_row
set mymasks    = ( 5152 502 261640 262142 5124 4 5120 ) # kernel selector: full_only split_only all 2,10,12 2 10,12
set mults      = "100"
set work_dist  = "1" # 1 approx eq NZE / 2 fancy NZE / else eq NRows
set thisexe    = "./HybridSpMVM_TUNE_SETUP_REVBUF_NLDD_OPENCL_PLACE_CYCLES_INT_dirac.x"
#set thismpi    = "/apps/OpenMPI/1.4.3-intel11.1.073-nomm/bin/mpirun" 
set thismpi    = "mpirun"
set thispin    = "/apps/likwid/devel/bin/likwid-pin"
set master     = `head -n 1 ${PBS_NODEFILE}`
set skip       = "-s 0x7"
set io_format  = "2" # 1 MPI-parallel / 2 binary / else ASCII
set outerit    = "1"
set datadir    = "./test-openmpi-111208-3kernels"
set wd_flag    = $distscheme[$work_dist]

mkdir $datadir
setenv KMP_AFFINITY disabled 
setenv OMP_SCHEDULE static
foreach mat ( dlr1 dlr2 dlr3 rrze3 rrze3_vv ) 

foreach nodes ( 1 )

   foreach jobtype ( 5 6 )
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
      set root         = "${master}_${mat}_NZE_${nodes}x${PEpernode}x${threadsperPE}_${gridDim}x${blockDim}_${jobtype}.out"
	 set ivargs       = "-n ${np} -npernode 2 -npersocket 1" # 2 gpus per node
#	 set ivargs       = "-n ${np} -bynode" # 1 gpu per node
# set ivargs       = "-n ${np}"
      
      if ( $threadsperPE == 1 ) then 
         set pin2cores = "L:0"
      else
  	    set pin2cores = "L:0-${lastthread}" # `echo $pinlist[$jobtype] | sed -e 's/_/,/g'`
      endif
   
	set pinlikwid    = "${thispin} ${skip} -c ${pin2cores}"
#	set pinlikwid    = ""
      #set pinlikwid     = "env LD_PRELOAD=/apps/rrze/lib/ptoverride-ubuntu64.so" # "${thispin} ${skip} -c ${pin2cores} -t intel"

      set command       = "time ${thismpi} ${ivargs}  ${pinlikwid} ${fixed} ${kernelparam}"
#set command       = "time ${thismpi} ${ivargs} -bind-to-socket -bycore ${pinlikwid} ${fixed} ${kernelparam}"

      echo "#nodes:" ${nodes} "pinmask:" $pinlist[$jobtype]
      mpirun --version
      echo $command

      setenv OMP_NUM_THREADS ${threadsperPE}
	  setenv OMP_STACKSIZE 1024M
      setenv I_MPI_DEVICE rdssm 

      $command > ${datadir}/Performance_${root}
      #echo "${gridDim} ${blockDim}" >> perfmeasure.dat
      #grep "Kern No. " ${datadir}/Performance_${root} >> perfmeasure.dat
    end
    end
   end
end

end
