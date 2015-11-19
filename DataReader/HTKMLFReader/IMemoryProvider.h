// Represents a memory provider. All processing units should be allocated with this memory provider.

#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {
    class IMemoryProvider
    {
    public:
        virtual void* allocate(size_t elmentSize, size_t numberOfElements) = 0;
        virtual void deallocate(void* p) = 0;
        virtual ~IMemoryProvider() = 0 {}
    };
}}}