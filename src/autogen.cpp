#include "ghost/config.h"
#include "ghost/autogen.h"
#include "ghost/util.h"
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

ghost_error ghost_autogen_kacz_nvecs(int **nvecs, int *n, int chunkheight, int nshifts)
{
    string configurations = GHOST_AUTOGEN_KACZ;
    if (configurations[configurations.length()-1] != ';') {
        configurations += ";";
    }

    size_t pos = 0, commapos = 0;
    int cfg_chunkheight, cfg_nvecs, cfg_nshifts;
    string configuration;
    vector<int> nvecs_vec;

    while ((pos = configurations.find(";")) != string::npos) {
        configuration = configurations.substr(0, pos);

        commapos = configuration.find(",");
        string cfg_chunkheight_str = configuration.substr(0,commapos);

        if (!cfg_chunkheight_str.compare("*")) {
            cfg_chunkheight = chunkheight;
        } else {
            cfg_chunkheight = stoi(cfg_chunkheight_str);
        }
        configuration.erase(0, commapos + 1);
        
        commapos = configuration.find(",");
        string cfg_nvecs_str = configuration.substr(0,commapos);
        if (!cfg_nvecs_str.compare("*")) {
            configurations.erase(0, pos + 1);
            continue;
        } else {
            cfg_nvecs = stoi(cfg_nvecs_str);
        }
        configuration.erase(0, commapos + 1);

        string cfg_nshifts_str = configuration;
        if (!cfg_nshifts_str.compare("*")) {
            cfg_nshifts = nshifts;
        } else {
            cfg_nshifts = stoi(cfg_nshifts_str);
        }

        if (chunkheight == cfg_chunkheight && nshifts == cfg_nshifts) {
            nvecs_vec.push_back(cfg_nvecs);
        }

        configurations.erase(0, pos + 1);
    }

    sort(nvecs_vec.begin(),nvecs_vec.end());
    *n = nvecs_vec.size();
    ghost_malloc((void **)nvecs,sizeof(int)*(*n));

    int i;
    for (i=0; i<*n; i++) {
        (*nvecs)[i] = nvecs_vec[i];
    }

    return GHOST_SUCCESS;


}
