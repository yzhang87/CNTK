#pragma once

#include "../interfaces/IPacker.h"

class Packer : public IPacker
{
protected:
    Packer(std::shared_ptr<IMemoryProvider> provider, size_t minibatchSize, std::shared_ptr<ISequencer> source/*, PackerConfig inputs*/)
        : IPacker(provider, minibatchSize, source)
    {}

public:
    virtual ~Packer() = 0 {}
};