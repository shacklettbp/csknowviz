#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "core.hpp"

namespace RLpbr {
namespace vk {

class CudaImportedBuffer {
public:
    CudaImportedBuffer(const DeviceState &dev,
                       int cuda_id,
                       VkDeviceMemory mem,
                       uint64_t num_bytes);

    CudaImportedBuffer(const CudaImportedBuffer &) = delete;
    CudaImportedBuffer(CudaImportedBuffer &&);
    ~CudaImportedBuffer();

    void *getDevicePointer() const { return dev_ptr_; }

private:
    int ext_fd_;
    cudaExternalMemory_t ext_mem_;
    void *dev_ptr_;
};

DeviceUUID getUUIDFromCudaID(int cuda_id);

}
}
