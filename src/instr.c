#include "ghost/instr.h"

static char *ghost_instr_prefix = "";
static char *ghost_instr_suffix = "";

void ghost_instr_prefix_set(char *prefix)
{
    ghost_instr_prefix = prefix;

}

char *ghost_instr_prefix_get()
{
    return ghost_instr_prefix;
}

void ghost_instr_suffix_set(char *suffix)
{
    ghost_instr_suffix = suffix;

}

char *ghost_instr_suffix_get()
{
    return ghost_instr_suffix;

}
