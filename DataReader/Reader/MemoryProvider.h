#pragma once

#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

    // Memory provider. It injected by CNTK into reader. 
    // Should be used for allocating the input data provided by the reader.
    class MemoryProvider
    {
    public:
        virtual void* Alloc(size_t element, size_t numberOfElements) = 0;
        virtual void Free(void* ptr) = 0;

        virtual ~MemoryProvider() = 0 {}
    };

    typedef std::shared_ptr<MemoryProvider> MemoryProviderPtr;
}}}
