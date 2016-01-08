#pragma once

#include <vector>
#include <memory>
#include "Sequences.h"
#include "DataTensor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Epoch configuration. Is communicated to the packer, randomizer, bundler.
    struct EpochConfiguration
    {
        size_t workerRank;
        size_t numberOfWorkers;

        size_t minibatchSize;
        size_t totalSize;

        size_t index;
    };

    typedef size_t InputId;

    typedef std::shared_ptr<ImageLayout> TensorShapePtr;

    // TODO: support more / extensible types?
    enum InputType
    {
        it_feature,
        it_label
    };

    // TODO: We can support more types in the future.
    enum ElementType
    {
        et_float,  // single precision
        et_double, // double precision
        et_atom    // sizeof(atom) == 1 constitute of blobs -> sequences of atoms.
    };

    // TODO: Other types, change the name.
    enum StorageType
    {
        st_dense,
        st_sparse_csc,
    };

    // Input description.
    struct InputDescription
    {
        std::wstring name;
        InputId id;
        InputType type;
        StorageType storageType;
        ElementType elementType;

        TensorShapePtr sampleLayout;
    };
    typedef std::shared_ptr<InputDescription> InputDescriptionPtr;

    // We introduced input type and
    // that essentially describes the representation of a particular element type
    // Also input type that describes the stream/input
    struct Layout
    {
        MBLayoutPtr columns;
        TensorShapePtr rows;
    };
    typedef std::shared_ptr<Layout> LayoutPtr;

    // Input data.
    // TODO: possibly get the data and size one function call.
    // Possibly change this to Stream - because it will be more network alligned
    class Input
    {
        void* data_;
        size_t data_size_;
        LayoutPtr layout_;

    public:
        Input(void* data, size_t dataSize, LayoutPtr layout)
            : data_(data)
            , data_size_(dataSize)
            , layout_(layout)
        {
        }

        const void* GetData() const
        {
            return data_;
        }

        size_t GetDataSize() const
        {
            return data_size_;
        }

        LayoutPtr GetLayout() const
        {
            return layout_;
        }
    };
    typedef std::shared_ptr<Input> InputPtr;

    // Memory provider. Should be used for allocating storage according to the Layout.
    class MemoryProvider
    {
    public:
        virtual void* Alloc(size_t element, size_t numberOfElements) = 0;
        virtual void Free(void* ptr) = 0;
        virtual ~MemoryProvider() = 0 {}
    };
    typedef std::shared_ptr<MemoryProvider> MemoryProviderPtr;

    // Represents a single minibatch.
    struct Minibatch
    {
        bool atEndOfEpoch;
        std::vector<InputPtr> minibatch;

        operator bool() const
        {
            return !atEndOfEpoch;
        }
    };

    // Main Reader interface. The border interface between the CNTK and Reader.
    class Reader
    {
    public:
        virtual std::vector<InputDescriptionPtr> GetInputs() = 0;
        virtual void StartEpoch(const EpochConfiguration& config) = 0;
        virtual Minibatch ReadMinibatch() = 0;
        virtual ~Reader() = 0 {};
    };
    typedef std::shared_ptr<Reader> ReaderPtr;
}}}
