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
            const std::vector<InputDescriptionPtr>& inputs);

        Minibatch ReadMinibatch();

    private:
        std::shared_ptr<void> AllocateBuffer(size_t numElements, size_t elementSize);

        MemoryProviderPtr m_memoryProvider;
        TransformerPtr m_transformer;
        std::vector<InputDescriptionPtr> m_inputs;
        std::vector<std::shared_ptr<void>> m_inputBuffers;

        MBLayoutPtr m_minibatchLayout;
        size_t m_mbSize;
        size_t m_elementSize;
    };

    typedef std::shared_ptr<FrameModePacker> FrameModePackerPtr;
}}}
