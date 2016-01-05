//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "ImageReaderNew.h"
#include "commandArgUtil.h"
#include "ImageTransformers.h"
#include "BlockRandomizer.h"
#include "ImageDataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    static bool AreEqual(const std::string& s1, const std::string& s2)
    {
        return std::equal(s1.begin(), s1.end(), s2.begin(), [](const char& a, const char& b) { return std::tolower(a) == std::tolower(b); });
    }

    ImageReaderNew::EpochImplementation::EpochImplementation(ImageReaderNew* parent)
        : m_parent(parent)
    {}

    ImageReaderNew::EpochImplementation::~EpochImplementation()
    {}

    Minibatch ImageReaderNew::EpochImplementation::ReadMinibatch()
    {
        return m_parent->GetMinibatch();
    }

    ImageReaderNew::ImageReaderNew(
        const ConfigParameters& parameters,
        size_t elementSize)
        : m_elementSize(elementSize), m_seed(0)
    {
        InitFromConfig(parameters);
    }

    void ImageReaderNew::InitFromConfig(const ConfigParameters& config)
    {
        DataDeserializerPtr deserializer = std::make_shared<ImageDataDeserializer>(config, m_elementSize);

        std::string rand = config(L"randomize", "auto");
        if (!AreEqual(rand, "auto"))
        {
            RuntimeError("Only Auto is currently supported.");
        }
        TransformerPtr randomizer = std::make_shared<BlockRandomizer>(0, SIZE_MAX, deserializer);

        auto inputs = deserializer->GetInputs();
        assert(inputs.size() == 2);
        auto features = std::find_if(inputs.begin(), inputs.end(), [](const InputDescriptionPtr& input) { return input->name == L"features"; });
        assert(features != inputs.end());
        auto labels = std::find_if(inputs.begin(), inputs.end(), [](const InputDescriptionPtr& input) { return input->name == L"labels"; });
        assert(labels != inputs.end());

        m_featDim = (*features)->sampleLayout->GetNumElements();
        m_labDim = (*labels)->sampleLayout->GetNumElements();

        TransformerPtr cropper = std::make_shared<CropTransformNew>(randomizer, (*features)->name, config((*features)->name), m_seed);
        TransformerPtr scaler = std::make_shared<ScaleTransform>(cropper, (*features)->name, m_seed, m_elementSize == 4 ? CV_32F : CV_64F, config((*features)->name));
        TransformerPtr mean = std::make_shared<MeanTransform>(scaler, (*features)->name);
        m_transformer = mean;
    }

    std::vector<InputDescriptionPtr> ImageReaderNew::GetInputs()
    {
        return m_transformer->GetInputs();
    }

    EpochPtr ImageReaderNew::StartNextEpoch(const EpochConfiguration& config)
    {
        assert(config.minibatchSize > 0);
        assert(config.totalSize > 0);

        m_transformer->SetEpochConfiguration(config);

        // TODO: move inside the shim.
        m_endOfEpoch = false;
        m_mbSize = config.minibatchSize;

        m_featBuf.resize(m_mbSize * m_featDim * m_elementSize);
        m_labBuf.resize(m_mbSize * m_labDim * m_elementSize);

        return std::make_shared<EpochImplementation>(this);
    }

    Minibatch ImageReaderNew::GetMinibatch()
    {
        assert(m_mbSize > 0);

        // TODO: should be in the shim?
        if (m_endOfEpoch)
        {
            Minibatch m;
            m.atEndOfEpoch = true;
            return m;
        }

        std::fill(m_labBuf.begin(), m_labBuf.end(), 0);

        Minibatch m;
        m.atEndOfEpoch = false;

        // TODO: Check that data deserializer and transformers are thread safe.
        //#pragma omp parallel for ordered schedule(dynamic)
        size_t mbSize = 0;
        for (size_t i = 0; i < m_mbSize; i++)
        {
            auto image = m_transformer->GetNextSequence();
            if(image.m_endOfEpoch)
            {
                m_endOfEpoch = true;
                break;
            }
            mbSize++;

            // features
            std::copy(
                reinterpret_cast<char*>(image.m_data[0].data),
                reinterpret_cast<char*>(image.m_data[0].data) + m_featDim * m_elementSize,
                m_featBuf.begin() + m_featDim * m_elementSize * i);

            // labels
            std::copy(
                reinterpret_cast<char*>(image.m_data[1].data),
                reinterpret_cast<char*>(image.m_data[1].data) + m_labDim * m_elementSize,
                m_labBuf.begin() + m_labDim * m_elementSize * i);
        }

        // Features
        LayoutPtr featureLayout = std::make_shared<Layout>();
        featureLayout->rows = std::make_shared<ImageLayout>(std::vector<size_t> { m_featDim });
        featureLayout->columns = std::make_shared<MBLayout>();
        featureLayout->columns->Init(mbSize, 1);
        InputPtr features = std::make_shared<Input>(&m_featBuf[0], m_featBuf.size(), featureLayout);
        m.minibatch.insert(std::make_pair(0, features));

        LayoutPtr labelLayout = std::make_shared<Layout>();
        labelLayout->rows = std::make_shared<ImageLayout>(std::vector<size_t> { m_labDim });
        labelLayout->columns = std::make_shared<MBLayout>();
        labelLayout->columns->Init(mbSize, 1);
        InputPtr labels = std::make_shared<Input>(&m_featBuf[0], m_featBuf.size(), labelLayout);
        m.minibatch.insert(std::make_pair(1, labels));
        return m;
    }
}}}
