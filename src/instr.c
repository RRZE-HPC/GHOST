#include "ghost/instr.h"
#include "ghost/func_util.h"

static char *ghost_instr_prefix = "";
static char *ghost_instr_suffix = "";
int ghost_instr_enable = 0;

void ghost_instr_prefix_set(const char *prefix)
{
    // function enter/exit macros would cause infinite recursion here
    ghost_instr_prefix = (char *)prefix;
}

char *ghost_instr_prefix_get()
{
    // function enter/exit macros would cause infinite recursion here
    return ghost_instr_prefix;
}

void ghost_instr_suffix_set(const char *suffix)
{
    ghost_instr_suffix = (char *)suffix;

}

char *ghost_instr_suffix_get()
{
    return ghost_instr_suffix;
}
