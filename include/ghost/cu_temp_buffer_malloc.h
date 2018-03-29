#pragma once
#include "ghost/error.h"


/**
     * @brief Useful for allocating small temporary buffers. Keeps a list of previously allocated
     * and freed buffers, and returns one if the size fits. Threadsafe.
     *
     * @param mem Where to store the memory.
     * @param bytesize The number of bytes to allocate.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
ghost_error ghost_cu_temp_buffer_malloc(void **mem, size_t bytesize);

/**
     * @brief Frees memory allocated with ghost_cu_temp_buffer_malloc.  Threadsafe.
     *
     * @param mem Which address to free
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
ghost_error ghost_cu_temp_buffer_free(void *mem);
