#include "stdafx.h"
#include "ImageDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    ImageDataDeserializer::ImageDataDeserializer(const ConfigParameters& /* config */)
    {
    }

    std::vector<InputDescriptionPtr> ImageDataDeserializer::GetInputs() const
    {
        std::vector<InputDescriptionPtr> dummy;
        return dummy;
    }

    void ImageDataDeserializer::SetEpochConfiguration(const EpochConfiguration& /* config */)
    {
    }

    const TimelineP& ImageDataDeserializer::GetSequenceDescriptions() const
    {
        return m_sequences;
    }

    std::vector<Sequence> ImageDataDeserializer::GetSequenceById(size_t /* id */)
    {
        std::vector<Sequence> dummy;
        return dummy;
    }

    bool ImageDataDeserializer::RequireChunk(size_t /* chunkIndex */)
    {
        return true;
    }

    void ImageDataDeserializer::ReleaseChunk(size_t /* chunkIndex */)
    {
    }

}}}