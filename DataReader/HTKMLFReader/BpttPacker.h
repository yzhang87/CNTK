#pragma once

#include "Packer.h"
#include <memory>
#include "IMemoryProvider.h"
#include "ISequencer.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    class BpttPacker : public Packer
    {
    public:
        BpttPacker(std::shared_ptr<IMemoryProvider> provider, size_t minibatchSize, std::shared_ptr<ISequencer> source/*, PackerConfig inputs*/)
            : Packer(provider, minibatchSize, source)
        {}
    };
}}}
