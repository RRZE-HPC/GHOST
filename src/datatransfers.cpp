#include "ghost/datatransfers.h"
#include "ghost/util.h"

#include <map>
#include <vector>
#include <string>
#include <numeric>
#include <sstream>
#include <ostream>
#include <iomanip>

using namespace std;

/**
 * @brief Region datatransfer accumulator
 */
typedef struct
{
    /**
     * @brief The sent bytes.
     */
    map<int,vector<size_t>> sendbytes;
    /**
     * @brief The sent bytes.
     */
    map<int,vector<size_t>> recvbytes;
    /**
     * @brief The last start time of this region.
     */
    double start;
} 
ghost_datatransfer_region_accu_t;

static map<string,ghost_datatransfer_region_accu_t> datatransfers;

ghost_error_t ghost_datatransfer_register(const char *tag, ghost_datatransfer_direction_t dir, int rank, size_t volume)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (dir == GHOST_DATATRANSFER_OUT) {
        datatransfers[tag].sendbytes[rank].push_back(volume);
    } else {
        datatransfers[tag].recvbytes[rank].push_back(volume);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

static size_t ghost_datatransfer_volume_get(const char *tag, ghost_datatransfer_direction_t dir, int rank)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    size_t vol = 0;

    if (rank == GHOST_DATATRANSFER_RANK_ALL || rank == GHOST_DATATRANSFER_RANK_ALL_W_GPU) {
        map<int,vector<size_t>>::iterator senditer, recviter;
        
        if (dir == GHOST_DATATRANSFER_OUT || dir == GHOST_DATATRANSFER_ANY) {
            for (senditer = datatransfers[tag].sendbytes.begin(); senditer != datatransfers[tag].sendbytes.end(); senditer++) {
                if ((senditer->first != GHOST_DATATRANSFER_RANK_GPU) || (rank == GHOST_DATATRANSFER_RANK_ALL_W_GPU)) {
                    vol += accumulate(senditer->second.begin(),senditer->second.end(),0.);
                }
            }
        } 
        if (dir == GHOST_DATATRANSFER_IN || dir == GHOST_DATATRANSFER_ANY) {
            for (recviter = datatransfers[tag].recvbytes.begin(); recviter != datatransfers[tag].recvbytes.end(); recviter++) {
                if ((recviter->first != GHOST_DATATRANSFER_RANK_GPU) || (rank == GHOST_DATATRANSFER_RANK_ALL_W_GPU)) {
                    vol += accumulate(recviter->second.begin(),recviter->second.end(),0.);
                }
            }
        }
    } else {
        if (dir == GHOST_DATATRANSFER_OUT || dir == GHOST_DATATRANSFER_ANY) {
            vol = accumulate(datatransfers[tag].sendbytes[rank].begin(),datatransfers[tag].sendbytes[rank].end(),0.);
        } 
        if (dir == GHOST_DATATRANSFER_IN || dir == GHOST_DATATRANSFER_ANY) {
            vol += accumulate(datatransfers[tag].recvbytes[rank].begin(),datatransfers[tag].recvbytes[rank].end(),0.);
        }
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return vol;
}

static int ghost_datatransfer_nneigh_get(const char *tag, ghost_datatransfer_direction_t dir)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);

    if (dir == GHOST_DATATRANSFER_OUT) {
        map<int,vector<size_t>> tmp = datatransfers[tag].sendbytes;
        tmp.erase(GHOST_DATATRANSFER_RANK_GPU);
        return tmp.size();
    } 
    if (dir == GHOST_DATATRANSFER_IN) {
        map<int,vector<size_t>> tmp = datatransfers[tag].recvbytes;
        tmp.erase(GHOST_DATATRANSFER_RANK_GPU);
        return tmp.size();
    }
    if (dir == GHOST_DATATRANSFER_ANY) {
        map<int,vector<size_t>> tmp = datatransfers[tag].sendbytes;
        tmp.insert(datatransfers[tag].recvbytes.begin(),datatransfers[tag].recvbytes.end());
        tmp.erase(GHOST_DATATRANSFER_RANK_GPU);
        return tmp.size();
    }
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return 0;
} 

ghost_error_t ghost_datatransfer_summarystring(char **str)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);

    stringstream buffer;
    map<string,ghost_datatransfer_region_accu_t>::iterator iter;
    
    size_t maxRegionLen = 0;
    size_t maxCallsLen = 0;
    
    stringstream tmp;
    for (iter = datatransfers.begin(); iter != datatransfers.end(); ++iter) {
        tmp << iter->first.length();
        maxRegionLen = max(iter->first.length(),maxRegionLen);
        tmp.str("");

        int ncalls_out = 0, ncalls_in = 0;
        map<int,vector<size_t>>::iterator tmpiter;
        for (tmpiter = iter->second.sendbytes.begin(); tmpiter != iter->second.sendbytes.end(); ++tmpiter) {
            if (tmpiter->first != GHOST_DATATRANSFER_RANK_GPU) {
                ncalls_out += tmpiter->second.size();
            }
        }
        for (tmpiter = iter->second.recvbytes.begin(); tmpiter != iter->second.recvbytes.end(); ++tmpiter) {
            if (tmpiter->first != GHOST_DATATRANSFER_RANK_GPU) {
                ncalls_in += tmpiter->second.size();
            }
        }
        tmp << ncalls_in << "," << ncalls_out;
        maxCallsLen = max(maxCallsLen,tmp.str().length());
        tmp.str("");
        
    }
    if (maxCallsLen < 10) {
        maxCallsLen = 10;
    }

    buffer << left << setw(maxRegionLen+1) << "Region" << right << " | ";
    buffer << setw(maxCallsLen+3) << "#Trans_MPI | ";
    buffer << " #Neigh_out | ";
    buffer << "  #Neigh_in | ";
    buffer << " #Neigh_tot | ";
    buffer << "      V_out | ";
    buffer << "       V_in | ";
    buffer << "      V_tot | ";
    buffer << "  V_out,gpu | ";
    buffer << "   V_in,gpu | ";
    buffer << "  V_tot,gpu" << endl;
    buffer << string(maxRegionLen+maxCallsLen+4+9*14,'-') << endl;
    
    buffer.precision(2);
    for (iter = datatransfers.begin(); iter != datatransfers.end(); ++iter) {
        int ncalls_out = 0, ncalls_in = 0;
        map<int,vector<size_t>>::iterator tmpiter;
        for (tmpiter = iter->second.sendbytes.begin(); tmpiter != iter->second.sendbytes.end(); ++tmpiter) {
            if (tmpiter->first != GHOST_DATATRANSFER_RANK_GPU) {
                ncalls_out += tmpiter->second.size();
            }
        }
        for (tmpiter = iter->second.recvbytes.begin(); tmpiter != iter->second.recvbytes.end(); ++tmpiter) {
            if (tmpiter->first != GHOST_DATATRANSFER_RANK_GPU) {
                ncalls_in += tmpiter->second.size();
            }
        }

        stringstream callbuf;
        callbuf << ncalls_out << "," << ncalls_in;

        buffer << scientific << left << setw(maxRegionLen+1) << iter->first << " | " << right << setw(maxCallsLen) <<
            callbuf.str() << " | " << setw(11) <<
            ghost_datatransfer_nneigh_get(iter->first.c_str(),GHOST_DATATRANSFER_OUT) << " | " << setw(11) <<
            ghost_datatransfer_nneigh_get(iter->first.c_str(),GHOST_DATATRANSFER_IN) << " | " << setw(11) <<
            ghost_datatransfer_nneigh_get(iter->first.c_str(),GHOST_DATATRANSFER_ANY) << " | " << setw(11) << scientific <<  
            (double)ghost_datatransfer_volume_get(iter->first.c_str(),GHOST_DATATRANSFER_OUT,GHOST_DATATRANSFER_RANK_ALL) << " | " << setw(11) << 
            (double)ghost_datatransfer_volume_get(iter->first.c_str(),GHOST_DATATRANSFER_IN,GHOST_DATATRANSFER_RANK_ALL) << " | " << setw(11) << 
            (double)ghost_datatransfer_volume_get(iter->first.c_str(),GHOST_DATATRANSFER_ANY,GHOST_DATATRANSFER_RANK_ALL) << " | " << setw(11) << 
            (double)ghost_datatransfer_volume_get(iter->first.c_str(),GHOST_DATATRANSFER_OUT,GHOST_DATATRANSFER_RANK_GPU) << " | " << setw(11) << 
            (double)ghost_datatransfer_volume_get(iter->first.c_str(),GHOST_DATATRANSFER_IN,GHOST_DATATRANSFER_RANK_GPU) << " | " << setw(11) << 
            (double)ghost_datatransfer_volume_get(iter->first.c_str(),GHOST_DATATRANSFER_ANY,GHOST_DATATRANSFER_RANK_GPU) 
            
            << endl;


    }
    
    GHOST_CALL_RETURN(ghost_malloc((void **)str,buffer.str().length()+1));
    strcpy(*str,buffer.str().c_str());

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}
