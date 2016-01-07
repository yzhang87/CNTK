//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#define _CRT_SECURE_NO_WARNINGS

#include "FrameModePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    FrameModePacker::FrameModePacker(
        TransformerPtr transformer,
        size_t minibatchSize,
        size_t elementSize,
        const std::vector<InputDescriptionPtr>& inputs)
        : m_transformer(transformer)
        , m_mbSize(minibatchSize)
        , m_elementSize(elementSize)
        , m_inputs(inputs)
        , m_minibatchLayout(std::make_shared<MBLayout>())
    {
        for (const auto& input : inputs)
        {
            size_t dimensions = input->sampleLayout->GetNumElements() * m_elementSize;

            std::vector<char> tmp;
            tmp.resize(m_mbSize * dimensions, 0);
            m_inputBuffers.push_back(tmp);
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
                    m_inputBuffers[j].begin() + dimensions * i);
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
            // TODO: add element and storage type

            size_t dimensions = m_inputs[i]->sampleLayout->GetNumElements() * m_elementSize;
            InputPtr stream = std::make_shared<Input>(&m_inputBuffers[i][0], mbSize * dimensions, layout);
            m.minibatch.insert(std::make_pair(i, stream));
        }
        return m;
    }
}}}
