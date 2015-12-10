#pragma once

#include "InnerInterfaces.h"
#include "ScpParser.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class MLFDataDeserializer : DataDeserializer
    {
    public:
        MLFDataDeserializer(ScpParserPtr nameToId);

        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;

        virtual const Timeline& GetTimeline() const override;

        virtual InputDescriptionPtr GetInput() const override;

        virtual Sequence GetSequenceById(size_t id) override;

        virtual Sequence GetSampleById(size_t sequenceId, size_t sampleId) override;

        virtual bool RequireChunk(size_t chunkIndex) override;

        virtual void ReleaseChunk(size_t chunkIndex) override;

    };
}}}