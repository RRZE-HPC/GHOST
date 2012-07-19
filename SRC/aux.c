#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <matricks.h>
#include <time.h>
#include <timing.h>

double RecalFrequency(uint64 cycles4measurement, double old_clockfreq)
{
   int i;

   uint64 asm_cycles, asm_cyclecounter;
   double startTime, stopTime, ct;
   struct timespec delay = { 0, 800000000 }; /* calibration time: 800 ms */
   double estimated_time_it_took, true_time_it_took;
   double recalibrated_CPUFrequency;

   for (i=0; i< 2; i++)
   {
      for_timing_start_asm_( &asm_cyclecounter);
      timing( &startTime, &ct);

      nanosleep( &delay, NULL);

      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles );
      timing( &stopTime, &ct);

      asm_cycles = asm_cycles - cycles4measurement; 
      estimated_time_it_took = (1.0*asm_cycles)/old_clockfreq; 
      true_time_it_took = stopTime-startTime;
      recalibrated_CPUFrequency = (1.0*asm_cycles)/true_time_it_took;

      IF_DEBUG(1){
	 printf("Recalibrating CPU-Frequency: \n");
	 printf("target time                  [s]  : %g\n", 0.8);
	 printf("estimated by rtdsci          [s]  : %g\n", estimated_time_it_took);
	 printf("measured by get_time_of_day  [s]  : %g\n", true_time_it_took);
	 printf("recalibrated CPUfrequency   [GHz] : %g\n", recalibrated_CPUFrequency);
	 printf("-------------------------------------\n");
      }
   }

   return recalibrated_CPUFrequency;
}

int get_NUMA_info(int *size0, int *free0, int *size1, int *free1){

   FILE *output;

   static int init_get_NUMA_info = 1; 

   if (init_get_NUMA_info==1){

      /* Abfragen, ob ich ueberhaupt auf einer Maschine mit NUMA-Architektur bin */

      init_get_NUMA_info = 0;
   }
//#define HAVE_NUMA
/*#ifdef HAVE_NUMA

   output = popen("numactl --hardware | grep 'node 0 size' | awk '{print $4}'", "r");
   if ( fscanf(output, "%d\n", size0) == 0 ) printf("MYERROR retrieving NUMA-size node 0");
   pclose(output);
   output = popen("numactl --hardware | grep 'node 1 size' | awk '{print $4}'", "r");
   if ( fscanf(output, "%d\n", size1) == 0 ) printf("MYERROR retrieving NUMA-size node 0");
   pclose(output);
   output = popen("numactl --hardware | grep 'node 0 free' | awk '{print $4}'", "r");
   if ( fscanf(output, "%d\n", free0) == 0 ) printf("MYERROR retrieving NUMA-size node 0");
   pclose(output);
   output = popen("numactl --hardware | grep 'node 1 free' | awk '{print $4}'", "r");
   if ( fscanf(output, "%d\n", free1) == 0 ) printf("MYERROR retrieving NUMA-size node 0");
   pclose(output);
#endif
*/
   //printf("%d %d %d %d\n", *size0, *size1, *free0, *free1);

   return(0);
}

float myCpuClockFrequency()
{
   static float frequency = -1.;

   if (frequency < 0) {
      FILE *output = popen("grep 'cpu MHz' /proc/cpuinfo | awk -F : '{print $2}'", "r");
      frequency = 0.0;
      if ( fscanf(output, "%f\n", &frequency) == 0 ) 
	 printf("MYERROR getting CPU-frequency");
      pclose(output);
      /* convert to Hz */
      frequency *= 1000000;
   }
   return frequency;
}

double my_amount_of_mem()
{
   static double mem = -1.0;
   int ret; 

   if (mem < 0) {
      FILE *output = popen("grep 'MemTotal:' /proc/meminfo | awk '{print $(NF-1)}'", "r");
      mem = 0.0;
      if ( (ret = fscanf(output, "%lf\n", &mem)) == 0 ) 
	 printf("MYERROR reading out /proc/meminfo: %d\n", ret);;
      pclose(output);
      /* convert to Hz */
      mem *= 1024;
   }
   return mem;
}

unsigned long machname(char* mach_name)
{
   int ret; 

   FILE *output = popen("uname -a | awk '{print $2}'", "r");
   ret = fscanf(output, "%s\n", mach_name);
   ret = strlen(mach_name);
   pclose(output);
   return ret;
}

unsigned long thishost(char* hostname)
{
   int ret; 

   FILE *output = popen("hostname", "r");
   ret = fscanf(output, "%s\n", hostname);
   ret = strlen(hostname);
   pclose(output);
   return ret;
}


unsigned long kernelversion(char* kernel_version)
{
   int ret; 

   FILE *output = popen("uname -a | awk '{print $3}'", "r");
   ret = fscanf(output, "%s\n", kernel_version);
   ret = strlen(kernel_version);
   pclose(output);
   return ret;
}

unsigned long modelname(char* model_name)
{
   int ret; 

   FILE *output = popen("cat /proc/cpuinfo | grep 'model name' | sort -u | sed -e 's/.*://' | sed -e 's/@.*//' | sed -e 's/\\ //g'", "r");
   ret = fscanf(output, "%s\n", model_name);
   ret = strlen(model_name);
   pclose(output);
   return ret;
}

uint64 cachesize()
{
   int ret; 
   uint64 cache_size;

   FILE *output = popen("cat /proc/cpuinfo | grep 'cache size' | sort -u | awk '{print $4}'", "r");
   if ( (ret = fscanf(output, "%llu\n", &cache_size)) == 0 ) 
      printf("MYERROR reading out /proc/cpuinfo: %d\n", ret);
   pclose(output);
   return cache_size;
}

void tmpwrite_d(int filenumber, int number_of_elements, real *datenarray){

   FILE *TMPFILE;
   char tmpfilename[50];
   int i;

   sprintf(tmpfilename, "./quasifort.%i", filenumber);
   if ((TMPFILE = fopen(tmpfilename, "w"))==NULL){
      printf("Fehler beim Oeffnen von %s\n", tmpfilename);
      exit(1);
   }

   for (i=0; i < number_of_elements ; i++){
      fprintf(TMPFILE,"%i %lg\n", i, datenarray[i]);
   } 
   fclose(TMPFILE);

   return;
} 
void tmpwrite_i(int filenumber, int number_of_elements, int *datenarray, char *mystring){

   FILE *TMPFILE;
   char tmpfilename[50];
   int i;

   sprintf(tmpfilename, "./quasifort.%i", filenumber);
   if ((TMPFILE = fopen(tmpfilename, "w"))==NULL){
      printf("Fehler beim Oeffnen von %s\n", tmpfilename);
      exit(1);
   }

   for (i=0; i < number_of_elements ; i++){
      fprintf(TMPFILE,"%i %s %i\n", i, mystring, datenarray[i]);
   } 
   fclose(TMPFILE);

   return;
} 

void mypabort(char *s) {
   printf("MYPABORT - %s\n", s);
#ifdef PROFILE
   vmon_done();
#endif
   MPI_Abort(MPI_COMM_WORLD, 999);
}

void mypaborts(const char *s1, const char *s2) {
   int ierr, me;
   ierr= MPI_Comm_rank(MPI_COMM_WORLD, &me);
   printf("PE%d: MYPABORT - %s %s\n", me, s1, s2);
#ifdef PROFILE
   vmon_done();
#endif
   MPI_Abort(MPI_COMM_WORLD, 999);
}

void myabort(char *s) {
   printf("MYABORT - %s\n", s);
#ifdef PROFILE
   vmon_done();
#endif
   MPI_Abort(MPI_COMM_WORLD,999);
}

void myaborti(char *s, int num) {
   printf("MYABORT - %s %d\n", s, num);
#ifdef PROFILE
   vmon_done();
#endif
   exit(0);
}

void myabortf(char *s, float num) {
   printf("MYABORT - %s %f\n", s, num);
#ifdef PROFILE
   vmon_done();
#endif
   exit(0);
}

void myaborttf(char *s, int tnum, float num) {
   printf("MYABORT - Thread %d: %s %f\n", tnum, s, num);
#ifdef PROFILE
   vmon_done();
#endif
   exit(0);
}

