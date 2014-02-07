#ifndef GHOST_INSTR_H
#define GHOST_INSTR_H

#if GHOST_HAVE_INSTR_TIMING

#include "log.h"
#include "timing.h"

#define GHOST_INSTR_START(tag)\
    double __start_##tag;\
    ghost_error_t __err_##tag;\
    GHOST_CALL(ghost_wctime(&__start_##tag),__err_##tag);\

#define GHOST_INSTR_STOP(tag)\
    double __end_##tag;\
    GHOST_CALL(ghost_wctime(&__end_##tag),__err_##tag);\
    LOG(TIMING,ANSI_COLOR_BLUE, "%s: %e secs" ANSI_COLOR_RESET,#tag,__end_##tag-__start_##tag);

#elif GHOST_HAVE_INSTR_LIKWID

#include <likwid.h>
#define GHOST_INSTR_START(tag) {\
_Pragma("omp parallel")\
    LIKWID_MARKER_START(#tag);}
#define GHOST_INSTR_STOP(tag) {\
_Pragma("omp parallel")\
    LIKWID_MARKER_STOP(#tag);}

#else

#define GHOST_INSTR_START(tag)
#define GHOST_INSTR_STOP(tag)

#endif

#endif
