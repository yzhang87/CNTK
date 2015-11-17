#pragma once


#include "Packer.h"

class BpttPacker : public Packer
{
public:
    BpttPacker(std::shared_ptr<IMemoryProvider> provider, size_t minibatchSize, std::shared_ptr<ISequencer> source/*, PackerConfig inputs*/)
        : Packer(provider, minibatchSize, source)
    {}
};