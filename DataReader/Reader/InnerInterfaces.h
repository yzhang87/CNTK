#pragma once

#include <string>
#include <vector>
#include "ReaderInterfaces.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    // Defines identifier and length of a Sequence.
    struct SequenceDescription
    {
        size_t id;
        size_t numberOfSamples;
        size_t chunkId;
        bool isValid;
    };

    // Defines a sequences, which consists of sequences description and a number
    // of frames, which have the same encoding and are layed out in memory contiguously.
    struct Sequence
    {
        TensorShapePtr layout;
        size_t numberOfSamples;
        void* data;
    };

    // Low-level input interface (for file, network, etc.).
    // Memory buffers to fill data into are provided by the caller.
    class BlockReader
    {
    public:
        virtual void Get(char* buffer, size_t offset, size_t size) = 0;
        virtual ~BlockReader() = 0 {}
    };

    // Timeline specifies a vector of Sequence IDs and lengths.
    // This information is exposed by a Sequencer, e.g., to be used by the Randomizer.
    typedef std::vector<SequenceDescription> Timeline;

    typedef std::vector<const SequenceDescription*> TimelineP;

    // Interface to for structured reading from a single data source
    class DataDeserializer
    {
    public:
        virtual std::vector<InputDescriptionPtr> GetInputs() const = 0; // TODO will remove
        virtual void SetEpochConfiguration(const EpochConfiguration& config) = 0;

        virtual const TimelineP& GetSequenceDescriptions() const = 0;
        virtual std::vector<Sequence> GetSequenceById(size_t id) = 0;

        virtual bool RequireChunk(size_t chunkIndex) = 0;
        virtual void ReleaseChunk(size_t chunkIndex) = 0;

        virtual ~DataDeserializer() = 0 {};
    };

    typedef std::shared_ptr<DataDeserializer> DataDeserializerPtr;

    struct SequenceData
    {
        SequenceData() : m_endOfEpoch(false)
        {}

        std::vector<Sequence> m_data;
        bool m_endOfEpoch;

        SequenceData(SequenceData&& other)
            : m_data(std::move(other.m_data))
            , m_endOfEpoch(std::move(other.m_endOfEpoch))
        {
        }
    };

    // Provides Input descriptions and sequential access to sequences.
    class Transformer
    {
    public:
        virtual void SetEpochConfiguration(const EpochConfiguration& config) = 0;
        virtual ~Transformer() = 0 {}
        virtual SequenceData GetNextSequence() = 0;
    };

    typedef std::shared_ptr<Transformer> TransformerPtr;
}}}
