#include "stdafx.h"
#include "HTKDataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    HTKDataDeserializer::HTKDataDeserializer(ScpParserPtr nameToId)
    {
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