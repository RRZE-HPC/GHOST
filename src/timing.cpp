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
    /**
     * @brief User-defined callback function to compute a region's performance.
     */
    ghost_compute_performance_func_t perfFunc;
    /**
     * @brief Argument to perfFunc.
     */
    void *perfFuncArg;
    /**
     * @brief The unit of performance.
     */
    const char *perfUnit;
} 
ghost_timing_region_accu_t;

static map<string,ghost_timing_region_accu_t> timings;

void ghost_timing_tick(const char *tag) 
{
    double start = 0.;
    ghost_timing_wc(&start);
  
    timings[tag].start = start;
}

void ghost_timing_tock(const char *tag) 
{
    double end;
    ghost_timing_wc(&end);
    ghost_timing_region_accu_t *ti = &timings[string(tag)];
    ti->times.push_back(end-ti->start);
}

void ghost_timing_set_perfFunc(const char *tag, ghost_compute_performance_func_t func, void *arg, const char *unit)
{
    timings[tag].perfFunc = func;
    timings[tag].perfFuncArg = arg;
    timings[tag].perfUnit = unit;
}


ghost_error_t ghost_timing_region_create(ghost_timing_region_t ** ri, const char *tag)
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
    (*ri)->accTime = accumulate(ti.times.begin(),ti.times.end(),0.);
    (*ri)->avgTime = (*ri)->accTime/(*ri)->nCalls;

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
    size_t maxUnitLen = 0;
    
    stringstream tmp;
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        int regLen = 0;
        if (iter->second.perfUnit) {
            tmp << iter->first.length();
            maxUnitLen = max(strlen(iter->second.perfUnit),maxUnitLen);
            regLen = strlen(iter->second.perfUnit);
            tmp.str("");
        }
        
        tmp << regLen+iter->first.length();
        maxRegionLen = max(regLen+iter->first.length(),maxRegionLen);
        tmp.str("");

        tmp << iter->second.times.size();
        maxCallsLen = max(maxCallsLen,tmp.str().length());
        tmp.str("");
        
    }
    if (maxCallsLen < 5) {
        maxCallsLen = 5;
    }

    buffer << left << setw(maxRegionLen+4) << "Region" << right << " | ";
    buffer << setw(maxCallsLen+3) << "Calls | ";
    buffer << "   t_min | ";
    buffer << "   t_max | ";
    buffer << "   t_avg | ";
    buffer << "   t_acc" << endl;
    buffer << string(maxRegionLen+maxCallsLen+7+4*11,'-') << endl;

    buffer.precision(2);
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        ghost_timing_region_t *region;
        ghost_timing_region_create(&region,iter->first.c_str());

        buffer << scientific << left << setw(maxRegionLen+4) << iter->first << " | " << right << setw(maxCallsLen) <<
            region->nCalls << " | " <<
            region->minTime << " | " <<
            region->maxTime << " | " <<
            region->avgTime << " | " <<
            region->accTime << endl;

        ghost_timing_region_destroy(region);
    }

    buffer << endl << endl << left << setw(maxRegionLen+4) << "Region" << right << " | ";
    buffer << setw(maxCallsLen+3) << "Calls | ";
    buffer << "   P_max | ";
    buffer << "   P_min | ";
    buffer << "   P_avg | ";
    buffer << "P_skip10" << endl;;
    buffer << string(maxRegionLen+maxCallsLen+7+4*11,'-') << endl;

    buffer.precision(2);
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        ghost_compute_performance_func_t pf = iter->second.perfFunc;
        if (!pf) {
            continue;
        }
        ghost_timing_region_t *region;
        ghost_timing_region_create(&region,iter->first.c_str());
        
        void *pfa = iter->second.perfFuncArg;

        double P_min = 0., P_max = 0., P_avg = 0., P_skip10 = 0.;
        int err = pf(&P_min,region->maxTime,pfa);
        if (err) {
            ERROR_LOG("Error in calling performance computation callback!");
        }
        err = pf(&P_max,region->minTime,pfa);
        if (err) {
            ERROR_LOG("Error in calling performance computation callback!");
        }
        err = pf(&P_avg,region->avgTime,pfa);
        if (err) {
            ERROR_LOG("Error in calling performance computation callback!");
        }
        if (region->nCalls > 10) {
            err = pf(&P_skip10,accumulate(iter->second.times.begin()+10,iter->second.times.end(),0.)/(region->nCalls-10),pfa);
            if (err) {
                ERROR_LOG("Error in calling performance computation callback!");
            }
        }

        buffer << scientific << left << setw(maxRegionLen-maxUnitLen+2) << iter->first << 
            right << "(" << setw(maxUnitLen) << iter->second.perfUnit << ")" << " | " << setw(maxCallsLen) <<
            region->nCalls << " | " <<
            P_max << " | " <<
            P_min << " | " <<
            P_avg << " | " <<
            P_skip10 << endl;
        
        ghost_timing_region_destroy(region);
    }


    GHOST_CALL_RETURN(ghost_malloc((void **)str,buffer.str().length()+1));
    strcpy(*str,buffer.str().c_str());

    return GHOST_SUCCESS;
}
