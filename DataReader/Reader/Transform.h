#pragma once

#include <vector>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ConfigParameters;

    // Defines a sequence.
    struct SequenceData
    {
        SequenceData() : m_endOfEpoch(false)
        {}

        // Data per stream. Id in the vector corresponds to the stream id returned from GetInputs.
        std::vector<Sequence> m_data;

        // End of epoch.
        bool m_endOfEpoch;

        SequenceData(SequenceData&& other)
            : m_data(std::move(other.m_data))
            , m_endOfEpoch(std::move(other.m_endOfEpoch))
        {
        }
    };

    class Transformer;
    typedef std::shared_ptr<Transformer> TransformerPtr;

    // Defines a data transformation interface.
    class Transformer
    {
    public:
        // Initialization.
        virtual void Initialize(
            TransformerPtr inputTransformer,
            const ConfigParameters& readerConfig,
            const std::vector<InputDescriptionPtr>& inputs) = 0;

        // Sets current epoch configuration.
        virtual void SetEpochConfiguration(const EpochConfiguration& config) = 0;

        // Gets next sequence.
        virtual SequenceData GetNextSequence() = 0;

        virtual ~Transformer() = 0 {}
    };
}}}
