#pragma once

#include "Packer.h"
#include "commandArgUtil.h" // for intargvector

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            class NormalPacker : public Packer
            {
            public:
                NormalPacker(std::shared_ptr<IMemoryProvider> provider, size_t minibatchSize, std::shared_ptr<ISequencer> source, const ConfigParameters inputs)
                    : Packer(provider, minibatchSize, source)
                {}
            };

        }
    }
}