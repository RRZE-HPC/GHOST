#include "ghost/cu_temp_buffer_malloc.h"
#include "ghost/error.h"
#include "ghost/log.h"
#include "ghost/util.h"
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <sstream>
#ifdef GHOST_HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace std;

struct SmallBuffer {
    size_t size = 0;
    void *dPtr = nullptr;
    bool used = false;
};

vector<SmallBuffer> buffers;
std::mutex cu_temp_buffer_malloc_mutex;
unsigned int usedBufferCount = 0;
unsigned int peakBufferCount = 0;

ghost_error ghost_cu_temp_buffer_malloc(void **mem, size_t bytesize)
{
#ifdef GHOST_HAVE_CUDA
    lock_guard<mutex> lock(cu_temp_buffer_malloc_mutex);


    usedBufferCount += 1;
    peakBufferCount = max(peakBufferCount, usedBufferCount);

    auto foundBuffer = find_if(begin(buffers), end(buffers),
        [=](const SmallBuffer &b) { return (b.size >= bytesize && !b.used); });

    if (foundBuffer != end(buffers)) {
        foundBuffer->used = true;
        *mem = foundBuffer->dPtr;
        GHOST_DEBUG_LOG(1, "Return %zuB buffer for %zuB Request", foundBuffer->size, bytesize);
    } else {
        SmallBuffer newBuffer;
        newBuffer.size = bytesize;
        newBuffer.used = true;
        CUDA_CALL_RETURN(cudaMalloc(&newBuffer.dPtr, bytesize));
        *mem = newBuffer.dPtr;
        buffers.push_back(newBuffer);

        sort(begin(buffers), end(buffers), [](const SmallBuffer &a, const SmallBuffer &b) {
            return (a.used == b.used && a.size < b.size) || (!a.used && b.used);
        });


        GHOST_DEBUG_LOG(1, "cudaMalloc new temporary buffer with  %zuB", bytesize);
        if (buffers.size() > peakBufferCount) {
            GHOST_DEBUG_LOG(1, "Have %zu buffers, needed %u at most, cudaFree buffer with %zuB",
                buffers.size(), peakBufferCount, begin(buffers)->size);
            CUDA_CALL_RETURN(cudaFree(begin(buffers)->dPtr));
            buffers.erase(begin(buffers));
        }
    }
#else
    UNUSED(mem);
    UNUSED(bytesize);
#endif

    return GHOST_SUCCESS;
}

ghost_error ghost_cu_temp_buffer_free(void *mem)
{
#ifdef GHOST_HAVE_CUDA
    lock_guard<mutex> lock(cu_temp_buffer_malloc_mutex);
    usedBufferCount -= 1;

    auto foundBuffer =
        find_if(begin(buffers), end(buffers), [=](SmallBuffer b) { return b.dPtr == mem; });
    if (foundBuffer != end(buffers)) {
        foundBuffer->used = false;
        GHOST_DEBUG_LOG(1, "Reclaimed %zuB buffer", foundBuffer->size);
    } else {
        GHOST_ERROR_LOG("Error, address  was not allocated with cu_temp_buffer_malloc\n");
        return GHOST_ERR_UNKNOWN;
    }

#else
    UNUSED(mem);
#endif
    return GHOST_SUCCESS;
}
