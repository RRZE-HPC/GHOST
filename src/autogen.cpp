#include "ghost/config.h"
#include "ghost/autogen.h"
#include "ghost/util.h"
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

static string autogen_str;
static int missingConfs = 0;

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
            cfg_chunkheight = atoi(cfg_chunkheight_str.c_str());
        }
        configuration.erase(0, commapos + 1);
        
        commapos = configuration.find(",");
        string cfg_nvecs_str = configuration.substr(0,commapos);
        if (!cfg_nvecs_str.compare("*")) {
            configurations.erase(0, pos + 1);
            continue;
        } else {
            cfg_nvecs = atoi(cfg_nvecs_str.c_str());
        }
        configuration.erase(0, commapos + 1);

        string cfg_nshifts_str = configuration;
        if (!cfg_nshifts_str.compare("*")) {
            cfg_nshifts = nshifts;
        } else {
            cfg_nshifts = atoi(cfg_nshifts_str.c_str());
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

ghost_error ghost_autogen_spmmv_nvecs(int **nvecs, int *n, int chunkheight)
{
    string configurations;
    ghost_type mytype;
    ghost_type_get(&mytype);

    if (mytype == GHOST_TYPE_CUDA) {
        configurations = GHOST_AUTOGEN_SPMMV_CUDA;
    } else {
        configurations = GHOST_AUTOGEN_SPMMV;
    }

    if (configurations[configurations.length()-1] != ';') {
        configurations += ";";
    }

    size_t pos = 0, commapos = 0;
    int cfg_chunkheight, cfg_nvecs;
    string configuration;
    vector<int> nvecs_vec;

    while ((pos = configurations.find(";")) != string::npos) {
        configuration = configurations.substr(0, pos);

        commapos = configuration.find(",");
        string cfg_chunkheight_str = configuration.substr(0,commapos);

        if (!cfg_chunkheight_str.compare("*")) {
            cfg_chunkheight = chunkheight;
        } else {
            cfg_chunkheight = atoi(cfg_chunkheight_str.c_str());
        }
        configuration.erase(0, commapos + 1);
        
        commapos = configuration.find(",");
        string cfg_nvecs_str = configuration.substr(0,commapos);
        if (!cfg_nvecs_str.compare("*")) {
            configurations.erase(0, pos + 1);
            continue;
        } else {
            cfg_nvecs = atoi(cfg_nvecs_str.c_str());
        }
        if (chunkheight == cfg_chunkheight) {
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

int ghost_autogen_spmmv_next_nvecs(int desired_nvecs, int chunkheight)
{
    int *nvecs = NULL;
    int n,i,found_nvecs=0;

    if (ghost_autogen_spmmv_nvecs(&nvecs,&n,chunkheight) != GHOST_SUCCESS) {
        return 0;
    }
    if (n == 0) {
        free(nvecs);
        return 0;
    }
    for (i=n-1; i>=0; i--) {
        if (nvecs[i] <= desired_nvecs) {
            found_nvecs = nvecs[i];
            break;
        }
    }

    free(nvecs);

    return found_nvecs;
}

ghost_error ghost_autogen_string_add(const char * func, const char * par)
{
    size_t funcpos = autogen_str.find(func);
    if (funcpos == string::npos) {
        autogen_str.append(" -DGHOST_AUTOGEN_");
        autogen_str.append(func);
        autogen_str.append("=\"");
        autogen_str.append(par);
        autogen_str.append("\"");
    } else {
        string funcstr = autogen_str.substr(funcpos+strlen(func)+1,autogen_str.find(" ",funcpos));
        string findpar_inner = ";"+string(par)+";";
        string findpar_last = ";"+string(par)+"\"";
        string findpar_first = "\""+string(par)+";";
        string findpar_only = "\""+string(par)+"\"";
        if ((funcstr.find(findpar_inner) == string::npos) 
                && (funcstr.find(findpar_last) == string::npos)
                && (funcstr.find(findpar_first) == string::npos)
                && (funcstr.find(findpar_only) == string::npos)) {
            autogen_str.insert(funcpos+strlen(func)+2,string(par)+";");
        }
    }

    return GHOST_SUCCESS;
}

const char *ghost_autogen_string()
{
    return autogen_str.c_str();
}

void ghost_autogen_set_missing()
{
    missingConfs = 1;
}

int ghost_autogen_missing()
{
    return missingConfs;
}
