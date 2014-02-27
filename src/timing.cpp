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

ghost_error_t ghost_timing_regionInfo_create(ghost_timing_regionInfo_t ** ri, char *region)
{
    ghost_timing_info_t ti;
    if (!timings.count(string(region))) {
        ERROR_LOG("The region %s does not exist!",region);
        return GHOST_ERR_INVALID_ARG;
    }
    ti = timings[string(region)];

    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_malloc((void **)ri,sizeof(ghost_timing_regionInfo_t)),err,ret);
    (*ri)->nCalls =  ti.secs.size();
    (*ri)->minTime = *min_element(ti.secs.begin(),ti.secs.end());
    (*ri)->maxTime = *max_element(ti.secs.begin(),ti.secs.end());
    (*ri)->avgTime = accumulate(ti.secs.begin(),ti.secs.end(),0.)/(*ri)->nCalls;

    GHOST_CALL_GOTO(ghost_malloc((void **)(&((*ri)->times)),sizeof(double)*(*ri)->nCalls),err,ret);
    memcpy((*ri)->times,&ti.secs[0],(*ri)->nCalls*sizeof(double));

    goto out;

err:
    ERROR_LOG("Freeing region info");
    if (*ri) {
        free((*ri)->times); (*ri)->times = NULL;
    }
    free(*ri); (*ri) = NULL;
out:

    return ret;
}

void ghost_timing_regionInfo_destroy(ghost_timing_regionInfo_t * ri)
{
    if (ri) {
        free(ri->times); ri->times = NULL;
    }
    free(ri);
}


ghost_error_t ghost_timing_summaryString(char **str)
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
