#include "ghost/instr.h"

char *ghost_instr_prefix = "";
char *ghost_instr_suffix = "";

void ghost_instr_setPrefix(char *prefix)
{
    ghost_instr_prefix = prefix;

}

void ghost_instr_setSuffix(char *suffix)
{
    ghost_instr_suffix = suffix;

}
