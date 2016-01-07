//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "ReaderInterfaces.h"
#include "InnerInterfaces.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class FrameModePacker
    {
        TransformerPtr m_transformer;
        std::vector<InputDescriptionPtr> m_inputs;
        std::vector<std::vector<char>> m_inputBuffers;

        MBLayoutPtr m_minibatchLayout;
        size_t m_mbSize;
        size_t m_elementSize;

    public:
        FrameModePacker(
            TransformerPtr transformer,
            size_t minibatchSize,
            size_t elementSize,
            const std::vector<InputDescriptionPtr>& inputs);

        Minibatch ReadMinibatch();
    };

    typedef std::shared_ptr<FrameModePacker> FrameModePackerPtr;
}}}
