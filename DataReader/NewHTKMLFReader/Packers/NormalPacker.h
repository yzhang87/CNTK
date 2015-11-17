#pragma once

#include "Packer.h"

class NormalPacker : public Packer
{
public:
    NormalPacker(std::shared_ptr<IMemoryProvider> provider, size_t minibatchSize, std::shared_ptr<ISequencer> source/*, PackerConfig inputs*/)
        : Packer(provider, minibatchSize, source)
    {}
};