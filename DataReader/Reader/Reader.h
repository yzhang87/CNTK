#pragma once

#include <vector>
#include <memory>
#include "Sequences.h"
#include "DataTensor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    struct ImageLayout;
    typedef std::shared_ptr<ImageLayout> TensorShapePtr;

    struct MBLayout;
    typedef std::shared_ptr<MBLayout> MBLayoutPtr;

    // Configuration for the current epoch.
    // Each time the epoch is started CNTK should communicate the configuration to the reader.
    struct EpochConfiguration
    {
        size_t numberOfWorkers;     // Number of the Open MPI workers for the current epoch.
        size_t workerRank;          // Rank of the Open MPI worker, worker rank has to be less the the number of workers.
        size_t minibatchSize;       // Minibatch size for the epoch.
        size_t totalSize;           // Total size of the epoch in samples.
        size_t index;               // Current epoch index [0 .. max number of epochs).
    };

    // Supported primitive element types.
    enum class ElementType
    {
        et_float,  // single precision
        et_double, // double precision
        et_atom    // sizeof(atom) == 1 constitute of blobs -> sequences of atoms (i.e. used for lattices, hmmm, etc.)
    };

    // Supported storage types.
    enum class StorageType
    {
        st_dense, // dense
    };

    // Type of streams.
    // TODO: should be deleted. This should be part of BS config.
    enum class InputType
    {
        it_feature,     // feature stream
        it_label,       // label stream
        it_opaque       // opaque stream
    };


    typedef size_t InputId;

    // This class describes a particular input: its name, elements, storage, etc.
    struct InputDescription
    {
        std::wstring name;              // Name of the input
        InputId id;                     // Id of the input
        InputType type;                 // Input type
        StorageType storageType;        // Storage type
        ElementType elementType;        // Element type
        TensorShapePtr sampleLayout;    // Layout of the sample for the input
    };
    typedef std::shared_ptr<InputDescription> InputDescriptionPtr;

    // Describes minibatch layout.
    struct Layout
    {
        MBLayoutPtr columns;
    };
    typedef std::shared_ptr<Layout> LayoutPtr;

    // Input data.
    // TODO: change it to Stream - because it will be more network alligned
    struct Input
    {
        void* data;
        size_t dataSize;
        LayoutPtr layout;           // Layout out of the data.
    };
    typedef std::shared_ptr<Input> InputPtr;

    // Represents a single minibatch, that contains information about all streams.
    struct Minibatch
    {
        bool atEndOfEpoch;                  // Signifies the end of the epoch.
        std::vector<InputPtr> minibatch;    // Minibatch data.

        Minibatch() : atEndOfEpoch(false)
        {}

        Minibatch(Minibatch&& other)
            : atEndOfEpoch(std::move(other.atEndOfEpoch))
            , minibatch(std::move(other.minibatch))
        {}
    };

    // Main Reader interface. The border interface between the CNTK and Reader.
    class Reader
    {
    public:
        // Describes the inputs this reader produces.
        virtual std::vector<InputDescriptionPtr> GetInputs() = 0;

        // Starts a new epoch.
        virtual void StartEpoch(const EpochConfiguration& config) = 0;

        // Reads a minibatch that contains data across all streams.
        virtual Minibatch ReadMinibatch() = 0;

        virtual ~Reader() = 0 {};
    };

    typedef std::shared_ptr<Reader> ReaderPtr;
}}}
