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

        TransformerPtr randomizer = std::make_shared<BlockRandomizer>(0, SIZE_MAX, deserializer);

        auto inputs = configHelper->GetInputs();
        assert(inputs.size() == 2);
        const auto & features = inputs[configHelper->GetFeatureInputIndex()];

        TransformerPtr cropper = std::make_shared<CropTransform>(randomizer, features->id, config(features->name), m_seed);
        TransformerPtr scaler = std::make_shared<ScaleTransform>(cropper, features->id, m_seed, m_elementSize == 4 ? CV_32F : CV_64F, config(features->name));
        TransformerPtr mean = std::make_shared<MeanTransform>(scaler, features->id);
        m_transformer = mean;
    }

    std::vector<InputDescriptionPtr> ImageReader::GetInputs()
    {
        return m_transformer->GetInputs();
        // TODO replace by configHelper->GetInputs();
    }

    void ImageReader::StartEpoch(const EpochConfiguration& config)
    {
        assert(config.minibatchSize > 0);
        assert(config.totalSize > 0);

        m_transformer->SetEpochConfiguration(config);
        m_packer = std::make_shared<FrameModePacker>(m_transformer, config.minibatchSize, m_elementSize, m_transformer->GetInputs() /* TODO */);
    }

    Minibatch ImageReader::ReadMinibatch()
    {
        return m_packer->ReadMinibatch();
    }
}}}
