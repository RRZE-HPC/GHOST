#define _GNU_SOURCE
#include "ghost/util.h"
#include "ghost/log.h"
#include "ghost/constants.h"

#include <stdarg.h>
#include <stdlib.h>
#include <errno.h>

//#define PRETTYPRINT

#define PRINTWIDTH 80
#define LABELWIDTH 40
#define HEADERHEIGHT 3
#define FOOTERHEIGHT 1

#ifdef PRETTYPRINT
#define PRINTSEP "┊"
#else
#define PRINTSEP ":"
#endif

#define VALUEWIDTH (PRINTWIDTH-LABELWIDTH-(int)strlen(PRINTSEP))
#define MAXVALUELEN 1024


ghost_error ghost_header_string(char **str, const char *fmt, ...)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);

    size_t headerLen = (PRINTWIDTH+1)*HEADERHEIGHT+1;
    *str = realloc(*str,strlen(*str)+headerLen);
    if (!(*str)) {
        GHOST_ERROR_LOG("Error in realloc");
        return GHOST_ERR_UNKNOWN;
    }
    memset(*str+strlen(*str),'\0',1);

    char label[1024] = "";

    if (fmt != NULL) {
        va_list args;
        va_start(args,fmt);
        vsnprintf(label,1024,fmt,args);
        va_end(args);
    }

    const int spacing = 4;
    int offset = 0;
    int len = strlen(label);
    int nDash = (PRINTWIDTH-2*spacing-len)/2;
    int rem = (PRINTWIDTH-2*spacing-len)%2;
    int i;
#ifdef PRETTYPRINT
    sprintf(*str,"┌");
    for (i=0; i<PRINTWIDTH-2; i++) sprintf(*str,"─");
    sprintf(*str,"┐");
    sprintf(*str,"\n");
    sprintf(*str,"├");
    for (i=0; i<nDash-1; i++) sprintf(*str,"─");
    for (i=0; i<spacing; i++) sprintf(*str," ");
    sprintf(*str,"%s",label);
    for (i=0; i<spacing+rem; i++) sprintf(*str," ");
    for (i=0; i<nDash-1; i++) sprintf(*str,"─");
    sprintf(*str,"┤");
    sprintf(*str,"\n");
    sprintf(*str,"├");
    for (i=0; i<LABELWIDTH; i++) sprintf(*str,"─");
    sprintf(*str,"┬");
    for (i=0; i<VALUEWIDTH; i++) sprintf(*str,"─");
    sprintf(*str,"┤");
    sprintf(*str,"\n");
#else
    for (i=0; i<PRINTWIDTH; i++) sprintf(*str+offset+i,"-");
    offset = PRINTWIDTH;
    sprintf(*str+offset,"\n");
    offset++;
    for (i=0; i<nDash; i++) sprintf(*str+offset+i,"-");
    offset += nDash;
    for (i=0; i<spacing; i++) sprintf(*str+offset+i," ");
    offset += spacing;
    sprintf(*str+offset,"%s",label);
    offset += strlen(label);
    for (i=0; i<spacing+rem; i++) sprintf(*str+offset+i," ");
    offset += spacing+rem;
    for (i=0; i<nDash; i++) sprintf(*str+offset+i,"-");
    offset += nDash;
    sprintf(*str+offset,"\n");
    offset++;
    for (i=0; i<PRINTWIDTH; i++) sprintf(*str+offset+i,"-");
    offset += PRINTWIDTH;
    sprintf(*str+offset,"\n");
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_footer_string(char **str) 
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);

    size_t len = strlen(*str);
    size_t footerLen = (PRINTWIDTH+1)*FOOTERHEIGHT+1;
    *str = realloc(*str,strlen(*str)+footerLen);
    if (!(*str)) {
        GHOST_ERROR_LOG("Error in realloc");
        return GHOST_ERR_UNKNOWN;
    }
    int i;
#ifdef PRETTYPRINT
    sprintf(*str,"└");
    for (i=0; i<LABELWIDTH; i++) sprintf(*str,"─");
    sprintf(*str,"┴");
    for (i=0; i<VALUEWIDTH; i++) sprintf(*str,"─");
    sprintf(*str,"┘");
#else
    for (i=0; i<PRINTWIDTH; i++) sprintf(*str+len+i,"-");
#endif
    sprintf(*str+len+PRINTWIDTH,"\n");

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_line_string(char **str, const char *label, const char *unit, const char *fmt, ...)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    int nLines, l;
    size_t len = strlen(*str);

    va_list args;
    va_start(args,fmt);
    char dummy[MAXVALUELEN+1];
    vsnprintf(dummy,MAXVALUELEN,fmt,args);
    va_end(args);
    nLines = 1+strlen(dummy)/VALUEWIDTH;

    // extend the string by PRINTWIDTH characters plus \n for each line plus \0
    *str = realloc(*str,len+nLines*(PRINTWIDTH+1)+1);
    if (!(*str)) {
        GHOST_ERROR_LOG("Error in realloc");
        return GHOST_ERR_UNKNOWN;
    }

    //memset(*str+len,'\0',1);

#ifdef PRETTYPRINT
    sprintf(*str,"│");
#endif
    if (unit) {
        int unitLen = strlen(unit);
        sprintf(*str+len,"%-*s (%s)%s%*s",LABELWIDTH-unitLen-3,label,unit,PRINTSEP,VALUEWIDTH,dummy);
    } else {
        sprintf(*str+len,"%-*s%s%*s",LABELWIDTH,label,PRINTSEP,VALUEWIDTH,dummy);
    }
    sprintf(*str+len+PRINTWIDTH,"\n");
    for (l=1; l<nLines; l++) {
        sprintf(*str+len+l*(PRINTWIDTH+1),"%-*s%s%*s",LABELWIDTH,"",PRINTSEP,VALUEWIDTH,dummy+l*VALUEWIDTH);
        //        sprintf(*str+len+l*(PRINTWIDTH+1),"%*s",PRINTWIDTH,dummy+l*VALUEWIDTH);
        sprintf(*str+len+(l+1)*PRINTWIDTH+1,"\n");
    }

#ifdef PRETTYPRINT
    sprintf(*str+len,"│");
#endif
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}


ghost_error ghost_malloc(void **mem, const size_t size)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (!mem) {
        GHOST_ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    if (size/(1024.*1024.*1024.) > 1.) {
        GHOST_DEBUG_LOG(1,"Allocating big array of size %f GB",size/(1024.*1024.*1024.));
    }

    *mem = malloc(size);

    if(!(*mem)) {
        GHOST_ERROR_LOG("Malloc of %zu bytes failed: %s",size,strerror(errno));
        return GHOST_ERR_UNKNOWN;
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_malloc_pinned(void **mem, const size_t size)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#ifdef GHOST_HAVE_CUDA
    return ghost_cu_malloc_pinned(mem,size);
#else
    return ghost_malloc(mem,size);
#endif
}

ghost_error ghost_malloc_align(void **mem, const size_t size, const size_t align)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);

    int ierr;
    if (size/(1024.*1024.*1024.) > 1.) {
        GHOST_DEBUG_LOG(1,"Allocating big array of size %f GB",size/(1024.*1024.*1024.));
    }

    if ((ierr = posix_memalign(mem, align, size)) != 0) {
        GHOST_ERROR_LOG("Error while allocating %zu bytes using posix_memalign: %s",size,strerror(ierr));
        return GHOST_ERR_UNKNOWN;
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

/*
// make this inline for not wasting too much time hashing for kernel-map indexes
int ghost_hash(int a, int b, int c)
{
GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);

a -= b; a -= c; a ^= (c>>13);
b -= c; b -= a; b ^= (a<<8);
c -= a; c -= b; c ^= (b>>13);
a -= b; a -= c; a ^= (c>>12);
b -= c; b -= a; b ^= (a<<16);
c -= a; c -= b; c ^= (b>>5);
a -= b; a -= c; a ^= (c>>3);
b -= c; b -= a; b ^= (a<<10);
c -= a; c -= b; c ^= (b>>15);

GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
return c;
}
*/

