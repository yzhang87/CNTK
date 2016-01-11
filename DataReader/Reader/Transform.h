#pragma once

#include <vector>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ConfigParameters;

    // Defines a sequence.
    struct Sequences
    {
        Sequences() : m_endOfEpoch(false)
        {}

        // Data per stream. Id in the outer vector have to corresponds to the stream id provided in the Initialize.
        std::vector<std::vector<SequenceData>> m_data;

        // End of epoch.
        bool m_endOfEpoch;

        Sequences(Sequences&& other)
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
            const ConfigParameters& readerConfig) = 0;

        // Describes streams the transorm produces.
        virtual std::vector<InputDescriptionPtr> GetInputs() const = 0;

        // Sets current epoch configuration.
        virtual void SetEpochConfiguration(const EpochConfiguration& config) = 0;

        // Gets next sequences.
        // The return value can be used till the next call to GetNextSequences.
        virtual Sequences GetNextSequences(size_t count) = 0;

        virtual ~Transformer() = 0 {}
    };
}}}
