//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include "FrameModePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    FrameModePacker::FrameModePacker(
        MemoryProviderPtr memoryProvider,
        TransformerPtr transformer,
        size_t minibatchSize,
        size_t elementSize,
        const std::vector<InputDescriptionPtr>& inputs)
        : m_transformer(transformer)
        , m_mbSize(minibatchSize)
        , m_elementSize(elementSize)
        , m_inputs(inputs)
        , m_minibatchLayout(std::make_shared<MBLayout>())
        , m_memoryProvider(memoryProvider)
    {
        for (const auto& input : inputs)
        {
            m_inputBuffers.push_back(AllocateBuffer(m_mbSize * input->sampleLayout->GetNumElements(), m_elementSize));
        }
    }

    Minibatch FrameModePacker::ReadMinibatch()
    {
        assert(m_mbSize > 0);

        Minibatch m;
        m.atEndOfEpoch = false;

        size_t mbSize = 0;
        for (size_t i = 0; i < m_mbSize; i++)
        {
            auto image = m_transformer->GetNextSequence();
            if (image.m_endOfEpoch)
            {
                m.atEndOfEpoch = true;
                break;
            }
            mbSize++;

            assert(m_inputBuffers.size() == image.m_data.size());
            for (int j = 0; j < image.m_data.size(); ++j)
            {
                size_t dimensions = m_inputs[j]->sampleLayout->GetNumElements() * m_elementSize;
                std::copy(
                    reinterpret_cast<char*>(image.m_data[j].data),
                    reinterpret_cast<char*>(image.m_data[j].data) + dimensions,
                    reinterpret_cast<char*>(m_inputBuffers[j].get()) + dimensions * i);
            }
        }

        if (mbSize == 0)
        {
            return m;
        }

        m_minibatchLayout->Init(mbSize, 1);
        for (int i = 0; i < m_inputs.size(); ++i)
        {
            LayoutPtr layout = std::make_shared<Layout>();
            layout->rows = m_inputs[i]->sampleLayout;
            layout->columns = m_minibatchLayout;
            size_t dimensions = m_inputs[i]->sampleLayout->GetNumElements() * m_elementSize;
            InputPtr stream = std::make_shared<Input>();
            stream->data = m_inputBuffers[i].get();
            stream->dataSize = mbSize * dimensions;
            stream->layout = layout;
            m.minibatch.push_back(stream);
        }

        return m;
    }

    std::shared_ptr<void> FrameModePacker::AllocateBuffer(size_t numElements, size_t elementSize)
    {
        return std::shared_ptr<void>(
            m_memoryProvider->Alloc(elementSize, numElements),
            [this](void* p) { m_memoryProvider->Free(p); });
    }
}}}
