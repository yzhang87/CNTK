#pragma once

#include "InnerInterfaces.h"
#include "commandArgUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageDataDeserializer : public DataDeserializer
    {
        TimelineP m_sequences;
    public:
        ImageDataDeserializer(const ConfigParameters& config); // TODO more

        std::vector<InputDescriptionPtr> GetInputs() const override;
        void SetEpochConfiguration(const EpochConfiguration& config) override;
        const TimelineP& GetSequenceDescriptions() const override;
        std::vector<Sequence> GetSequenceById(size_t id) override;
        bool RequireChunk(size_t chunkIndex) override;
        void ReleaseChunk(size_t chunkIndex) override;
    };
}}}