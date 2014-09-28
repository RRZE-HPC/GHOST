#include "ghost/instr.h"

static char *ghost_instr_prefix = "";
static char *ghost_instr_suffix = "";

void ghost_instr_prefix_set(const char *prefix)
{
    ghost_instr_prefix = (char *)prefix;

}

char *ghost_instr_prefix_get()
{
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
