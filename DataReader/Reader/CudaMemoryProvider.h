//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include <memory>
#include <CUDAPageLockedMemAllocator.h>

#include "MemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class CudaMemoryProvider : public MemoryProvider
    {
        std::unique_ptr<CUDAPageLockedMemAllocator> m_allocator;

    public:
        CudaMemoryProvider(int deviceId)
        {
            m_allocator = std::make_unique<CUDAPageLockedMemAllocator>(deviceId);
        }

        virtual void* Alloc(size_t elementSize, size_t numberOfElements) override
        {
            size_t totalSize = elementSize * numberOfElements;
            return m_allocator->Malloc(totalSize);
        }

        virtual void Free(void* p) override
        {
            if (!p)
            {
                return;
            }

            m_allocator->Free(reinterpret_cast<char*>(p));
        }
    };

}}}
