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
        MemoryProviderPtr provider,
        const ConfigParameters& config,
        ElementType elementType)
        : m_provider(provider)
        , m_seed(0)
        , m_elementType(elementType)
    {
        // In the future, deserializers and transformers will be dynamically loaded 
        // from external libraries based on the configuration/brain script.
        // We will provide ability to implement the transformer and 
        // deserializer interface not only in C++ but in scripting languages as well.

        auto configHelper = ImageConfigHelper(config);
        m_streams = configHelper.GetStreams();
        assert(m_streams.size() == 2);
        DataDeserializerPtr deserializer = std::make_shared<ImageDataDeserializer>(config);
        TransformerPtr randomizer = std::make_shared<BlockRandomizer>(0, SIZE_MAX, deserializer);

        TransformerPtr cropper = std::make_shared<CropTransform>();
        TransformerPtr scaler = std::make_shared<ScaleTransform>();
        TransformerPtr mean = std::make_shared<MeanTransform>();

        cropper->Initialize(randomizer, config);
        scaler->Initialize(cropper, config);
        mean->Initialize(scaler, config);
        m_transformer = mean;
    }

    std::vector<StreamDescriptionPtr> ImageReader::GetStreams()
    {
        assert(!m_streams.empty());
        return m_streams;
    }

    void ImageReader::StartEpoch(const EpochConfiguration& config)
    {
        assert(config.minibatchSize > 0);
        assert(config.totalSize > 0);

        m_transformer->SetEpochConfiguration(config);
        m_packer = std::make_shared<FrameModePacker>(
            m_provider,
            m_transformer,
            config.minibatchSize,
            m_elementType == ElementType::et_float ? sizeof(float) : sizeof(double),
            m_streams);
    }

    Minibatch ImageReader::ReadMinibatch()
    {
        assert(m_packer != nullptr);
        return m_packer->ReadMinibatch();
    }
}}}
