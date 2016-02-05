#include "ghost/instr.h"
#include "ghost/func_util.h"

//static char *ghost_instr_prefix = "";
//static char *ghost_instr_suffix = "";
//int ghost_instr_enable = 0;

static pthread_key_t ghost_instr_prefix_key;
static pthread_key_t ghost_instr_suffix_key;
pthread_key_t ghost_instr_enable_key;

ghost_error ghost_instr_create()
{
    int err = 0;
    err += pthread_key_create(&ghost_instr_prefix_key,NULL);
    err += pthread_key_create(&ghost_instr_suffix_key,NULL);
    err += pthread_key_create(&ghost_instr_enable_key,NULL);
    
    if (err) {
        ERROR_LOG("An error occured!");
        return GHOST_ERR_UNKNOWN;
    }

    return GHOST_SUCCESS;
}

ghost_error ghost_instr_destroy()
{
    int err = 0;
    err += pthread_key_delete(ghost_instr_prefix_key);
    err += pthread_key_delete(ghost_instr_suffix_key);
    err += pthread_key_delete(ghost_instr_enable_key);

    if (err) {
        ERROR_LOG("An error occured!");
        return GHOST_ERR_UNKNOWN;
    }

    return GHOST_SUCCESS;
}

void ghost_instr_prefix_set(const char *prefix)
{
    // function enter/exit macros would cause infinite recursion here
    pthread_setspecific(ghost_instr_prefix_key,prefix);
}

char *ghost_instr_prefix_get()
{
    // function enter/exit macros would cause infinite recursion here
    if (pthread_getspecific(ghost_instr_prefix_key)) {
        return (char *)pthread_getspecific(ghost_instr_prefix_key);
    } else {
        return "";
    }
}

void ghost_instr_suffix_set(const char *suffix)
{
    pthread_setspecific(ghost_instr_suffix_key,suffix);
}

char *ghost_instr_suffix_get()
{
    if (pthread_getspecific(ghost_instr_suffix_key)) {
        return (char *)pthread_getspecific(ghost_instr_suffix_key);
    } else {
        return "";
    }
}
