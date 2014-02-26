#include "ghost/timing.h"
#include "ghost/util.h"

#include <map>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>

using namespace std;

typedef struct
{
    vector<double> secs;
    double start;
} 
ghost_timing_info_t;

static map<string,ghost_timing_info_t> timings;

void ghost_timing_tick(char *tag) 
{
    double start;
    ghost_wctime(&start);
    timings[tag].start = start;
}

void ghost_timing_tock(char *tag) 
{
    double end;
    ghost_wctime(&end);
    ghost_timing_info_t *ti = &timings[string(tag)];
    ti->secs.push_back(end-ti->start);
}

ghost_error_t ghost_timing_string(char **str)
{
    stringstream buffer;
    map<string,ghost_timing_info_t>::iterator iter;

   
    size_t maxRegionLen = 0;
    size_t maxCallsLen = 0;
    
    stringstream tmp;
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        tmp << iter->first.length();
        maxRegionLen = max(iter->first.length(),maxRegionLen);
        tmp.str("");

        tmp << iter->second.secs.size();
        maxCallsLen = max(maxCallsLen,tmp.str().length());
        tmp.str("");
    }
    if (maxCallsLen < 5) {
        maxCallsLen = 5;
    }

    buffer << left << setw(maxRegionLen) << "Region" << right << " | ";
    buffer << setw(maxCallsLen+3) << "Calls | ";
    buffer << "   t_min | ";
    buffer << "   t_max | ";
    buffer << "   t_avg" << endl;
    buffer << string(maxRegionLen+maxCallsLen+3+3*11,'-') << endl;

    buffer.precision(2);
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        buffer << scientific << left << setw(maxRegionLen) << iter->first << " | " << 
            right << setw(maxCallsLen) << iter->second.secs.size() << " | " <<
            *min_element(iter->second.secs.begin(),iter->second.secs.end()) << " | " <<
            *max_element(iter->second.secs.begin(),iter->second.secs.end()) << " | " <<
            accumulate(iter->second.secs.begin(),iter->second.secs.end(),0.)/iter->second.secs.size() << endl;
    }



    GHOST_CALL_RETURN(ghost_malloc((void **)str,buffer.str().length()+1));
    strcpy(*str,buffer.str().c_str());

    return GHOST_SUCCESS;


}
