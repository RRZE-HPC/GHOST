#include "ghost/densemat.h"

ghost_densemat_flags_t operator|(ghost_densemat_flags_t a, ghost_densemat_flags_t b) 
{
    return static_cast<ghost_densemat_flags_t>(static_cast<int>(a) | static_cast<int>(b));
}

ghost_densemat_flags_t operator|=(ghost_densemat_flags_t a, ghost_densemat_flags_t b) 
{
    return static_cast<ghost_densemat_flags_t>(static_cast<int>(a) | static_cast<int>(b));
}
