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

        auto images = m_transformer->GetNextSequences(m_mbSize);

        if (images.m_endOfEpoch)
        {
            m.atEndOfEpoch = true;
        }

        for (size_t i = 0; i < images.m_data.size(); i++)
        {
            assert(m_inputBuffers.size() == images.m_data[i].size());
            for (int j = 0; j < images.m_data[i].size(); ++j)
            {
                size_t dimensions = m_inputs[j]->sampleLayout->GetNumElements() * m_elementSize;
                std::copy(
                    reinterpret_cast<char*>(images.m_data[i][j].data),
                    reinterpret_cast<char*>(images.m_data[i][j].data) + dimensions,
                    reinterpret_cast<char*>(m_inputBuffers[j].get()) + dimensions * i);
            }
        }

        if (images.m_data.size() == 0)
        {
            return m;
        }

        m_minibatchLayout->Init(images.m_data.size(), 1);
        for (int i = 0; i < m_inputs.size(); ++i)
        {
            size_t dimensions = m_inputs[i]->sampleLayout->GetNumElements() * m_elementSize;
            InputPtr stream = std::make_shared<Input>();
            stream->data = m_inputBuffers[i].get();
            stream->dataSize = images.m_data.size() * dimensions;
            stream->layout = m_minibatchLayout;

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
