#pragma once

#include "InnerInterfaces.h"
#include "ScpParser.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class MLFDataDeserializer : public DataDeserializer
    {
        size_t m_dimension;
        SampleLayoutPtr m_layout;
        std::wstring m_stateListPath;
        std::vector<std::wstring> m_mlfPaths;

        // [classidsbegin+t] concatenation of all state sequences
        msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE> m_classIds;

    public:
        MLFDataDeserializer(const ConfigParameters& label);

        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;

        virtual const Timeline& GetTimeline() const override;

        virtual InputDescriptionPtr GetInput() const override;

        virtual Sequence GetSequenceById(size_t id) override;

        virtual Sequence GetSampleById(size_t sequenceId, size_t sampleId) override;

        virtual bool RequireChunk(size_t chunkIndex) override;

        virtual void ReleaseChunk(size_t chunkIndex) override;

    };
}}}