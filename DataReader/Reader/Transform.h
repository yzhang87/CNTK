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

        // Data for up to a requested number of sequences.
        // Indices in the inner vector have to correspond to the stream IDs
        // given by GetStream().
        std::vector<std::vector<SequenceData>> m_data;

        // Indicates whether the epoch ends with the data returned.
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
            TransformerPtr next,
            const ConfigParameters& readerConfig) = 0;

        // Describes streams the transformer produces.
        virtual std::vector<StreamDescriptionPtr> GetStreams() const = 0;

        // Sets current epoch configuration.
        virtual void SetEpochConfiguration(const EpochConfiguration& config) = 0;

        // Gets next sequences.
        // The return value can be used until the next call to GetNextSequences.
        virtual Sequences GetNextSequences(size_t count) = 0;

        virtual ~Transformer() = 0 {}
    };
}}}
