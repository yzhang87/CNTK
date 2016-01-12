//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "Reader.h"
#include "MemoryProvider.h"
#include "Transform.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class FrameModePacker
    {
    public:
        FrameModePacker(
            MemoryProviderPtr memoryProvider,
            TransformerPtr transformer,
            size_t minibatchSize,
            size_t elementSize,
            const std::vector<StreamDescriptionPtr>& streams);

        Minibatch ReadMinibatch();

    private:
        std::shared_ptr<char> AllocateBuffer(size_t numElements, size_t elementSize);

        MemoryProviderPtr m_memoryProvider;
        TransformerPtr m_transformer;
        std::vector<StreamDescriptionPtr> m_outputStreams;
        std::vector<StreamDescriptionPtr> m_inputStreams;
        std::vector<std::shared_ptr<char>> m_streamBuffers;

        MBLayoutPtr m_minibatchLayout;
        size_t m_mbSize;
        size_t m_elementSize;
    };

    typedef std::shared_ptr<FrameModePacker> FrameModePackerPtr;
}}}
