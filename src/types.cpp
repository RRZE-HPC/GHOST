#include "ghost/types.h"

ghost_datatype_t operator|(ghost_datatype_t a, ghost_datatype_t b) 
{
    return static_cast<ghost_datatype_t>(static_cast<int>(a) | static_cast<int>(b));
}
