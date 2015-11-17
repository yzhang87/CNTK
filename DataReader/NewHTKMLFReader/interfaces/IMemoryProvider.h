// Represents a memory provider. All processing units should be allocated with this memory provider.

#pragma once

class IMemoryProvider
{
public:
    virtual char* allocate(size_t size) = 0;
    virtual void deallocate(char* p) = 0;
    virtual ~IMemoryProvider() = 0 {}
};
