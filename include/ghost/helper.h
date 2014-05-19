/**
 * @file helper.h
 * @brief Helpers for applications which use GHOST.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_HELPER_H
#define GHOST_HELPER_H

#define GHOST_MAIN_TASK_START \
    typedef struct\
{\
    int argc;\
    char **argv;\
}\
ghost_main_task_args;\
\
static void *ghost_main_task(void *varg) {\
    int argc = ((ghost_main_task_args *)varg)->argc;\
    char ** argv = ((ghost_main_task_args *)varg)->argv;


#define GHOST_MAIN_TASK_END \
    return NULL;\
}\
int main(int argc, char** argv) {\
    ghost_main_task_args arg;\
    arg.argc = argc;\
    arg.argv = argv;\
    ghost_init(argc,argv);\
    ghost_task_t *t;\
    ghost_task_create(&t,GHOST_TASK_FILL_ALL, 0, &ghost_main_task, &arg, GHOST_TASK_DEFAULT, NULL, 0);\
    ghost_task_enqueue(t);\
    ghost_task_wait(t);\
    ghost_task_destroy(t);\
    ghost_finalize();\
    return EXIT_SUCCESS;\
}

#endif
