#include "ghost/instr.h"

static char *ghost_instr_prefix = "";
static char *ghost_instr_suffix = "";

void ghost_instr_setPrefix(char *prefix)
{
    ghost_instr_prefix = prefix;

}

char *ghost_instr_getPrefix()
{
    return ghost_instr_prefix;
}

void ghost_instr_setSuffix(char *suffix)
{
    ghost_instr_suffix = suffix;

}

char *ghost_instr_getSuffix()
{
    return ghost_instr_suffix;

}
