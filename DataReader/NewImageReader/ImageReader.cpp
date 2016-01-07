//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "ImageReader.h"
#include "commandArgUtil.h"
#include "ImageConfigHelper.h"
#include "ImageTransformers.h"
#include "BlockRandomizer.h"
#include "ImageDataDeserializer.h"
#include "FrameModePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    static bool AreEqual(const std::string& s1, const std::string& s2)
    {
        return std::equal(s1.begin(), s1.end(), s2.begin(), [](const char& a, const char& b) { return std::tolower(a) == std::tolower(b); });
    }

    ImageReader::ImageReader(
        const ConfigParameters& parameters,
        size_t elementSize)
        : m_elementSize(elementSize), m_seed(0)
    {
        InitFromConfig(parameters);
    }

    void ImageReader::InitFromConfig(const ConfigParameters& config)
    {
        auto configHelper = std::make_shared<ImageConfigHelper>(config);
        DataDeserializerPtr deserializer = std::make_shared<ImageDataDeserializer>(configHelper, m_elementSize);

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

        TransformerPtr cropper = std::make_shared<CropTransform>(randomizer, (*features)->name, config((*features)->name), m_seed);
        TransformerPtr scaler = std::make_shared<ScaleTransform>(cropper, (*features)->name, m_seed, m_elementSize == 4 ? CV_32F : CV_64F, config((*features)->name));
        TransformerPtr mean = std::make_shared<MeanTransform>(scaler, (*features)->name);
        m_transformer = mean;
    }

    std::vector<InputDescriptionPtr> ImageReader::GetInputs()
    {
        return m_transformer->GetInputs();
    }

    void ImageReader::StartEpoch(const EpochConfiguration& config)
    {
        assert(config.minibatchSize > 0);
        assert(config.totalSize > 0);

        m_transformer->SetEpochConfiguration(config);
        m_packer = std::make_shared<FrameModePacker>(m_transformer, config.minibatchSize, m_elementSize, m_transformer->GetInputs());
    }

    Minibatch ImageReader::ReadMinibatch()
    {
        return m_packer->ReadMinibatch();
    }
}}}
