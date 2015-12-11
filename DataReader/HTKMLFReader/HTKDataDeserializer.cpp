#include "stdafx.h"
#include "HTKDataDeserializer.h"
#include "ConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    HTKDataDeserializer::HTKDataDeserializer(const ConfigParameters& feature)
        : m_featureFiles(std::move(ConfigHelper::GetFeaturePaths(feature)))
    {
        ConfigHelper::CheckFeatureType(feature);

        auto context = ConfigHelper::GetContextWindow(feature);

        m_dimension = feature(L"dim");
        m_dimension = m_dimension * (1 + context.first + context.second);

        m_layout = std::make_shared<ImageLayout>(std::move(std::vector<size_t>{ m_dimension }));
    }

    void Microsoft::MSR::CNTK::HTKDataDeserializer::SetEpochConfiguration(const EpochConfiguration& /*config*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    const Timeline& Microsoft::MSR::CNTK::HTKDataDeserializer::GetTimeline() const
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    Microsoft::MSR::CNTK::InputDescriptionPtr Microsoft::MSR::CNTK::HTKDataDeserializer::GetInput() const
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    Microsoft::MSR::CNTK::Sequence Microsoft::MSR::CNTK::HTKDataDeserializer::GetSequenceById(size_t /*id*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    Microsoft::MSR::CNTK::Sequence Microsoft::MSR::CNTK::HTKDataDeserializer::GetSampleById(size_t /*sequenceId*/, size_t /*sampleId*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    bool Microsoft::MSR::CNTK::HTKDataDeserializer::RequireChunk(size_t /*chunkIndex*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    void Microsoft::MSR::CNTK::HTKDataDeserializer::ReleaseChunk(size_t /*chunkIndex*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

}}}