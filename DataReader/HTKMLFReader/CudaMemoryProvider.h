#pragma once

#include <memory>
#include <CUDAPageLockedMemAllocator.h>

#include "IMemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class CudaMemoryProvider : public IMemoryProvider
    {
        std::unique_ptr<CUDAPageLockedMemAllocator> m_allocator;

    public:
        CudaMemoryProvider(int deviceId)
        {
            m_allocator = std::make_unique<CUDAPageLockedMemAllocator>(deviceId);
        }

        virtual void* allocate(size_t elmentSize, size_t numberOfElements) override
        {
            size_t totalSize = elmentSize * numberOfElements;
            return m_allocator->Malloc(totalSize);
        }

        virtual void deallocate(void* p) override
        {
            if (!p)
            {
                return;
            }

            m_allocator->Free(reinterpret_cast<char*>(p));
        }
    };
}}}
