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
    /**
     * @brief User-defined callback function to compute a region's performance.
     */
    ghost_compute_performance_func perfFunc;
    /**
     * @brief Argument to perfFunc.
     */
    void *perfFuncArg;
    /**
     * @brief The unit of performance.
     */
    const char *perfUnit;
}
ghost_timing_perfFunc;


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

    vector<ghost_timing_perfFunc> perfFuncs;
} 
ghost_timing_region_accu;

static map<string,ghost_timing_region_accu> timings;
static pthread_mutex_t timingsMutex = PTHREAD_MUTEX_INITIALIZER;

void ghost_timing_tick(const char *tag) 
{
    pthread_mutex_lock(&timingsMutex);

    double start = 0.;
    ghost_timing_wc(&start);
    timings[tag].start = start;
    
    pthread_mutex_unlock(&timingsMutex);
}

void ghost_timing_tock(const char *tag) 
{
    pthread_mutex_lock(&timingsMutex);
    double end;
    ghost_timing_wc(&end);
    ghost_timing_region_accu *ti = &timings[string(tag)];
    ti->times.push_back(end-ti->start);
    pthread_mutex_unlock(&timingsMutex);
}

void ghost_timing_set_perfFunc(const char *prefix, const char *tag, ghost_compute_performance_func func, void *arg, size_t sizeofarg, const char *unit)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
  
    if (!tag) {
        goto out;
    }

    ghost_timing_perfFunc pf;
    pf.perfFunc = func;
    pf.perfUnit = unit;

    
    ghost_timing_region_accu *regionaccu;
    if (prefix) {
        string fulltag = (string(prefix)+"->"+string(tag));
        regionaccu = &timings[fulltag];
    } else {
        regionaccu = &timings[string(tag)];
    }

    for (std::vector<ghost_timing_perfFunc>::iterator it = regionaccu->perfFuncs.begin();
            it !=regionaccu->perfFuncs.end(); ++it) {
        if (it->perfFunc == func && !strcmp(it->perfUnit,unit)) {
            goto out;
        }
    }

    ghost_malloc((void **)&(pf.perfFuncArg),sizeofarg);
    memcpy(pf.perfFuncArg,arg,sizeofarg);
    regionaccu->perfFuncs.push_back(pf);
    
out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
}


ghost_error ghost_timing_region_create(ghost_timing_region ** ri, const char *tag)
{
    ghost_timing_region_accu ti = timings[string(tag)];
    if (!ti.times.size()) {
        *ri = NULL;
        return GHOST_SUCCESS;
    }

    ghost_error ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_malloc((void **)ri,sizeof(ghost_timing_region)),err,ret);
    (*ri)->nCalls =  ti.times.size();
    (*ri)->minTime = *min_element(ti.times.begin(),ti.times.end());
    (*ri)->maxTime = *max_element(ti.times.begin(),ti.times.end());
    (*ri)->accTime = accumulate(ti.times.begin(),ti.times.end(),0.);
    (*ri)->avgTime = (*ri)->accTime/(*ri)->nCalls;
    if ((*ri)->nCalls > 10) {
        (*ri)->skip10avgTime = accumulate(ti.times.begin()+10,ti.times.end(),0.)/((*ri)->nCalls-10);
    } else {
        (*ri)->skip10avgTime = 0.;
    }


    GHOST_CALL_GOTO(ghost_malloc((void **)(&((*ri)->times)),sizeof(double)*(*ri)->nCalls),err,ret);
    memcpy((*ri)->times,&ti.times[0],(*ri)->nCalls*sizeof(double));

    goto out;

err:
    GHOST_ERROR_LOG("Freeing region info");
    if (*ri) {
        free((*ri)->times); (*ri)->times = NULL;
    }
    free(*ri); (*ri) = NULL;
out:

    return ret;
}

void ghost_timing_region_destroy(ghost_timing_region * ri)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    
    if (ri) {
        free(ri->times); ri->times = NULL;
    }
    free(ri);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
}

void ghost_timing_destroy()
{
    map<string,ghost_timing_region_accu>::iterator iter;
    vector<ghost_timing_perfFunc>::iterator pf_iter;
    
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        for (pf_iter = iter->second.perfFuncs.begin(); pf_iter != iter->second.perfFuncs.end(); ++pf_iter) {
            if (pf_iter->perfFuncArg != NULL) {
                free(pf_iter->perfFuncArg);
            }
            pf_iter->perfFuncArg = NULL;
        }
    }
    timings.clear();
}

ghost_error ghost_timing_summarystring(char **str)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    
    stringstream buffer;
    map<string,ghost_timing_region_accu>::iterator iter;
    vector<ghost_timing_perfFunc>::iterator pf_iter;

   
    size_t maxRegionLen = 0;
    size_t maxCallsLen = 0;
    size_t maxUnitLen = 0;
    
    stringstream tmp;
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        int regLen = 0;
        for (pf_iter = iter->second.perfFuncs.begin(); pf_iter != iter->second.perfFuncs.end(); ++pf_iter) {
            if (pf_iter->perfUnit) {
                tmp << iter->first.length();
                maxUnitLen = max(strlen(pf_iter->perfUnit),maxUnitLen);
                regLen = strlen(pf_iter->perfUnit);
                tmp.str("");
            }
            
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
    buffer << "   t_s10 | ";
    buffer << "   t_acc" << endl;
    buffer << string(maxRegionLen+maxCallsLen+7+5*11,'-') << endl;

    buffer.precision(2);
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        ghost_timing_region *region = NULL;
        ghost_timing_region_create(&region,iter->first.c_str());

        if (region) {
            buffer << scientific << left << setw(maxRegionLen+4) << iter->first << " | " << right << setw(maxCallsLen) <<
                region->nCalls << " | " <<
                region->minTime << " | " <<
                region->maxTime << " | " <<
                region->avgTime << " | " <<
                region->skip10avgTime << " | " <<
                region->accTime << endl;

            ghost_timing_region_destroy(region);
        }
    }

    int printed = 0;
    buffer.precision(2);
    for (iter = timings.begin(); iter != timings.end(); ++iter) {
        if (!printed) {
            buffer << endl << endl << left << setw(maxRegionLen+4) << "Region" << right << " | ";
            buffer << setw(maxCallsLen+3) << "Calls | ";
            buffer << "   P_max | ";
            buffer << "   P_min | ";
            buffer << "   P_avg | ";
            buffer << "P_skip10" << endl;;
            buffer << string(maxRegionLen+maxCallsLen+7+4*11,'-') << endl;
        }
        printed = 1;

        ghost_timing_region *region = NULL;
        ghost_timing_region_create(&region,iter->first.c_str());
        
        if (region) {
            for (pf_iter = iter->second.perfFuncs.begin(); pf_iter != iter->second.perfFuncs.end(); ++pf_iter) {
                ghost_compute_performance_func pf = pf_iter->perfFunc;
                void *pfa = pf_iter->perfFuncArg;

                double P_min = 0., P_max = 0., P_avg = 0., P_skip10 = 0.;
                int err = pf(&P_min,region->maxTime,pfa);
                if (err) {
                    GHOST_ERROR_LOG("Error in calling performance computation callback!");
                }
                err = pf(&P_max,region->minTime,pfa);
                if (err) {
                    GHOST_ERROR_LOG("Error in calling performance computation callback!");
                }
                err = pf(&P_avg,region->avgTime,pfa);
                if (err) {
                    GHOST_ERROR_LOG("Error in calling performance computation callback!");
                }
                if (region->nCalls > 10) {
                    err = pf(&P_skip10,accumulate(iter->second.times.begin()+10,iter->second.times.end(),0.)/(region->nCalls-10),pfa);
                    if (err) {
                        GHOST_ERROR_LOG("Error in calling performance computation callback!");
                    }
                }

                buffer << scientific << left << setw(maxRegionLen-maxUnitLen+2) << iter->first << 
                    right << "(" << setw(maxUnitLen) << pf_iter->perfUnit << ")" << " | " << setw(maxCallsLen) <<
                    region->nCalls << " | " <<
                    P_max << " | " <<
                    P_min << " | " <<
                    P_avg << " | " <<
                    P_skip10 << endl;
            }
            ghost_timing_region_destroy(region);
        }
    }


    GHOST_CALL_RETURN(ghost_malloc((void **)str,buffer.str().length()+1));
    strcpy(*str,buffer.str().c_str());

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}
