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

/**
 * @brief Region timing accumulator
 */
typedef struct
{
    /**
     * @brief The runtimes of this region.
     */
    vector<double> times;
    /**
     * @brief The last start time of this region.
     */
    double start;
} 
ghost_timing_region_accu_t;

static map<string,ghost_timing_region_accu_t> timings;

void ghost_timing_tick(char *tag) 
{
    double start;
    ghost_timing_wc(&start);
    timings[tag].start = start;
}

void ghost_timing_tock(char *tag) 
{
    double end;
    ghost_timing_wc(&end);
    ghost_timing_region_accu_t *ti = &timings[string(tag)];
    ti->times.push_back(end-ti->start);
}

ghost_error_t ghost_timing_region_create(ghost_timing_region_t ** ri, char *tag)
{
    ghost_timing_region_accu_t ti;
    if (!timings.count(string(tag))) {
        ERROR_LOG("The region %s does not exist!",tag);
        return GHOST_ERR_INVALID_ARG;
    }
    ti = timings[string(tag)];

    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_malloc((void **)ri,sizeof(ghost_timing_region_t)),err,ret);
    (*ri)->nCalls =  ti.times.size();
    (*ri)->minTime = *min_element(ti.times.begin(),ti.times.end());
    (*ri)->maxTime = *max_element(ti.times.begin(),ti.times.end());
    (*ri)->avgTime = accumulate(ti.times.begin(),ti.times.end(),0.)/(*ri)->nCalls;

    GHOST_CALL_GOTO(ghost_malloc((void **)(&((*ri)->times)),sizeof(double)*(*ri)->nCalls),err,ret);
    memcpy((*ri)->times,&ti.times[0],(*ri)->nCalls*sizeof(double));

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

void ghost_timing_region_destroy(ghost_timing_region_t * ri)
{
    if (ri) {
        free(ri->times); ri->times = NULL;
    }
    free(ri);
}


ghost_error_t ghost_timing_summarystring(char **str)
{
    stringstream buffer;
    map<string,ghost_timing_region_accu_t>::iterator iter;

   
    size_t maxRegionLen = 0;
    size_t maxCallsLen = 0;
    
    stringstream tmp;
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        tmp << iter->first.length();
        maxRegionLen = max(iter->first.length(),maxRegionLen);
        tmp.str("");

        tmp << iter->second.times.size();
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
    buffer << "   t_avg | ";
    buffer << "   t_tot" << endl;
    buffer << string(maxRegionLen+maxCallsLen+3+4*11,'-') << endl;

    buffer.precision(2);
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        buffer << scientific << left << setw(maxRegionLen) << iter->first << " | " << 
            right << setw(maxCallsLen) << iter->second.times.size() << " | " <<
            *min_element(iter->second.times.begin(),iter->second.times.end()) << " | " <<
            *max_element(iter->second.times.begin(),iter->second.times.end()) << " | " <<
            accumulate(iter->second.times.begin(),iter->second.times.end(),0.)/iter->second.times.size()  << " | " <<
            accumulate(iter->second.times.begin(),iter->second.times.end(),0.) << endl;
    }



    GHOST_CALL_RETURN(ghost_malloc((void **)str,buffer.str().length()+1));
    strcpy(*str,buffer.str().c_str());

    return GHOST_SUCCESS;
}
