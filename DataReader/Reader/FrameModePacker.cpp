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
        const std::vector<StreamDescriptionPtr>& streams)
        : m_transformer(transformer)
        , m_mbSize(minibatchSize)
        , m_elementSize(elementSize)
        , m_outputStreams(streams)
        , m_minibatchLayout(std::make_shared<MBLayout>())
        , m_memoryProvider(memoryProvider)
    {
        m_inputStreams = m_transformer->GetStreams();
        assert(m_inputStreams.size() == m_outputStreams.size());
        assert(
            std::find_if(
                m_outputStreams.begin(),
                m_outputStreams.end(), 
                [](const StreamDescriptionPtr& s) { return s->storageType == StorageType::st_sparse_csc; }) == m_outputStreams.end());

        for (const auto& stream : streams)
        {
            m_streamBuffers.push_back(AllocateBuffer(m_mbSize * stream->sampleLayout->GetNumElements(), m_elementSize));
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
            assert(m_streamBuffers.size() == images.m_data[i].size());
            for (int j = 0; j < images.m_data[i].size(); ++j)
            {
                assert(images.m_data[i][j].numberOfSamples == 1);

                size_t dimensions = m_inputStreams[j]->sampleLayout->GetNumElements() * m_elementSize;
                const char* source = reinterpret_cast<char*>(images.m_data[i][j].data);
                if (m_inputStreams[j]->storageType == StorageType::st_dense)
                {
                    std::copy(
                        source,
                        source + dimensions,
                        m_streamBuffers[j].get() + dimensions * i);
                }
                else if (m_inputStreams[j]->storageType == StorageType::st_sparse_csc)
                {
                    TensorShapePtr layout = images.m_data[i][j].layouts[0];
                    std::fill(m_streamBuffers[j].get() + i * dimensions, m_streamBuffers[j].get() + (i + 1) * dimensions, 0);
                    size_t nonZeroCount = layout->GetDim(0);
                    for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
                    {
                        size_t rowIndex = reinterpret_cast<size_t*>(images.m_data[i][j].data)[nonZeroIndex];
                        char* destination = m_streamBuffers[j].get() + dimensions * i + rowIndex * m_elementSize;
                        source = source + nonZeroCount * sizeof(size_t) + rowIndex * m_elementSize;
                        std::copy(source, source + m_elementSize, destination);
                    }
                }
                else
                {
                    RuntimeError("Storage type %d is not supported.", m_inputStreams[j]->storageType);
                }
            }
        }

        if (images.m_data.size() == 0)
        {
            return m;
        }

        m_minibatchLayout->Init(images.m_data.size(), 1);
        for (int i = 0; i < m_outputStreams.size(); ++i)
        {
            size_t dimensions = m_outputStreams[i]->sampleLayout->GetNumElements() * m_elementSize;
            StreamPtr stream = std::make_shared<Stream>();
            stream->data = m_streamBuffers[i].get();
            stream->dataSize = images.m_data.size() * dimensions;
            stream->layout = m_minibatchLayout;

            m.minibatch.push_back(stream);
        }

        return m;
    }

    std::shared_ptr<char> FrameModePacker::AllocateBuffer(size_t numElements, size_t elementSize)
    {
        return std::shared_ptr<char>(
            reinterpret_cast<char*>(m_memoryProvider->Alloc(elementSize, numElements)),
            [this](char* p) { m_memoryProvider->Free(p); });
    }
}}}
