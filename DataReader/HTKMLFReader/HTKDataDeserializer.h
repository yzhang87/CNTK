#pragma once

#include "InnerInterfaces.h"
#include "ScpParser.h"
#include "BundlerSplitted.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class HTKDataDeserializer : public DataDeserializer
    {
        struct HTKSequenceDescription : public SequenceDescription
        {
            HTKSequenceDescription(utterancedesc&& u) : utterance(u) {}

            utterancedesc utterance;
        };

        size_t m_dimension;
        SampleLayoutPtr m_layout;
        std::vector<std::wstring> m_featureFiles;

        std::vector<HTKSequenceDescription> m_sequences;
        TimelineP m_sequencesP;

    public:
        HTKDataDeserializer(const ConfigParameters& feature);

        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;

        virtual TimelineP GetSequenceDescriptions() const override;

        virtual InputDescriptionPtr GetInput() const override;

        virtual Sequence GetSequenceById(size_t id) override;

        virtual Sequence GetSampleById(size_t sequenceId, size_t sampleId) override;

        virtual bool RequireChunk(size_t chunkIndex) override;

        virtual void ReleaseChunk(size_t chunkIndex) override;
    };
}}}