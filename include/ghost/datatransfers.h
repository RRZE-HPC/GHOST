#include "error.h"

#define GHOST_DATATRANSFER_RANK_GPU -1
#define GHOST_DATATRANSFER_RANK_ALL -2
#define GHOST_DATATRANSFER_RANK_ALL_W_GPU -3

typedef enum {
    GHOST_DATATRANSFER_IN,
    GHOST_DATATRANSFER_OUT,
    GHOST_DATATRANSFER_ANY
} ghost_datatransfer_direction_t;

#ifdef __cplusplus
extern "C" {
#endif

    ghost_error_t ghost_datatransfer_register(const char *tag, ghost_datatransfer_direction_t dir, int rank, size_t volume);
    ghost_error_t ghost_datatransfer_summarystring(char **str);

#ifdef __cplusplus
}
#endif
