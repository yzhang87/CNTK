#pragma once

#include <vector>
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Defines properties of a sequence.
    // Randomization is based on this data structure.
    struct SequenceDescription
    {
        size_t id;                  // Sequence id
        size_t numberOfSamples;     // Number of samples in a sequence
        size_t chunkId;             // Each sequence belongs to an I/O chunk, how chunk is defined is specific to the data deserializer.
        bool isValid;
    };

    typedef std::vector<const SequenceDescription*> Timeline;

    // Defines sequence data and its layout.
    struct SequenceData
    {
        SequenceData() : data(nullptr) {}
        void* data;
    };

    typedef std::shared_ptr<SequenceData> SequenceDataPtr;

    struct DenseSequenceData : SequenceData
    {
        DenseSequenceData() : numberOfSamples(0) {}

        TensorShapePtr sampleLayout;
        size_t numberOfSamples; // Number of samples in the sequence
    };

    typedef std::shared_ptr<DenseSequenceData> DenseSequenceDataPtr;

    struct SparseSequenceData : SequenceData
    {
        std::vector<std::vector<size_t>> indices;
    };

    typedef std::shared_ptr<SparseSequenceData> SparseSequenceDataPtr;

    // Interface for reading data from several streams.
    class DataDeserializer
    {
    public:
        // Describes streams the data deserializer produces.
        virtual std::vector<StreamDescriptionPtr> GetStreams() const = 0;

        // Sets epoch configuration.
        virtual void SetEpochConfiguration(const EpochConfiguration& config) = 0;

        // Retrieve global timeline the data deserializer can produce.
        virtual const Timeline& GetSequenceDescriptions() const = 0;

        // Gets sequences by id.
        // The return value can be used until the next call to GetSequencesById.
        virtual std::vector<std::vector<SequenceDataPtr>> GetSequencesById(const std::vector<size_t> & ids) = 0;

        // Require chunk.
        virtual bool RequireChunk(size_t chunkIndex) = 0;

        // Release chunk.
        virtual void ReleaseChunk(size_t chunkIndex) = 0;

        virtual ~DataDeserializer() = 0 {};
    };

    typedef std::shared_ptr<DataDeserializer> DataDeserializerPtr;
}}}
