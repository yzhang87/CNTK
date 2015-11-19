#pragma once

#include "IMemoryProvider.h"
#include <memory>
#include <algorithm>

namespace Microsoft { namespace MSR { namespace CNTK {

    class HeapMemoryProvider : public IMemoryProvider
    {
        static const size_t size_of_first_pointer = sizeof(void*);

    public:
        virtual void* allocate(size_t elmentSize, size_t numberOfElements) override
        {
            size_t allignment = max(elmentSize, size_of_first_pointer);
            size_t request_size = elmentSize * numberOfElements + allignment;
            size_t needed = size_of_first_pointer + request_size;

            void* allocted = ::operator new(needed);
            void* allowed_space = reinterpret_cast<char*>(allocted) + size_of_first_pointer;
            void* p = std::align(allignment, elmentSize, allowed_space, request_size);

            // save for delete calls to use
            (reinterpret_cast<void**>(p))[-1] = allocted;
            return p;
        }

        virtual void deallocate(void* p) override
        {
            if (!p)
            {
                return;
            }

            void * alloc = reinterpret_cast<void**>(p)[-1];
            ::operator delete(alloc);
        }
    };
}}}
