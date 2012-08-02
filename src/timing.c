#include "timing.h"

void timing(double* wcTime, double* cpuTime)
{
#ifdef __sun
   hrtime_t t;

   t = gethrtime(); // time in nanoseconds
   *wcTime = ((double)t)/1.0e9;
   *cpuTime = 0.0;
#else
   struct timeval tp;
   struct rusage ruse;

   gettimeofday(&tp, NULL);
   *wcTime=(double) (tp.tv_sec + tp.tv_usec/1000000.0); 
  
   getrusage(RUSAGE_SELF, &ruse);
   *cpuTime=(double)(ruse.ru_utime.tv_sec+ruse.ru_utime.tv_usec / 1000000.0);
#endif
}
