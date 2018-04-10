#include "ghost/cu_temp_buffer_malloc.h"
#include "ghost/error.h"
#include "ghost/log.h"
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
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

ghost_error ghost_cu_temp_buffer_malloc(void **mem, size_t bytesize) {
  lock_guard<mutex> lock(cu_temp_buffer_malloc_mutex);

  auto foundBuffer =
      find_if(begin(buffers), end(buffers), [=](const SmallBuffer &b) {
        return (b.size >= bytesize && !b.used);
      });

  if (foundBuffer != end(buffers)) {
    foundBuffer->used = true;
    *mem = foundBuffer->dPtr;
    //    cout << "Return " << foundBuffer->size << " B buffer for " << bytesize
    //     << " B Request\n";
  } else {
    SmallBuffer newBuffer;
    newBuffer.size = bytesize;
    newBuffer.used = true;
    CUDA_CALL_RETURN(cudaMalloc(&newBuffer.dPtr, bytesize));
    *mem = newBuffer.dPtr;
    buffers.push_back(newBuffer);
    sort(begin(buffers), end(buffers),
         [](const SmallBuffer &a, const SmallBuffer &b) {
           return (!a.used && b.used) || a.size < b.size;
         });
    // cout << "Malloc " << bytesize << " bytes\n";
  }

  return GHOST_SUCCESS;
}

ghost_error ghost_cu_temp_buffer_free(void *mem) {
  lock_guard<mutex> lock(cu_temp_buffer_malloc_mutex);

  auto foundBuffer = find_if(begin(buffers), end(buffers),
                             [=](SmallBuffer b) { return b.dPtr == mem; });
  if (foundBuffer != end(buffers)) {
    foundBuffer->used = false;
    // cout << "Reclaimed " << foundBuffer->size << " B buffer\n";
  } else {
    GHOST_ERROR_LOG(
        "Error, address  was not allocated with cu_temp_buffer_malloc\n");
    return GHOST_ERR_UNKNOWN;
  }

  return GHOST_SUCCESS;
}
