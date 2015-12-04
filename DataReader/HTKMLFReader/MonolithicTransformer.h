#pragma once

#include <vector>
#include <memory>

#include "reader_interface.h"
#include "commandArgUtil.h"

#include "biggrowablevectors.h"
#include "utterancesourcemultiNew.h"
#include "minibatchiterator.h"
#include <inner_interfaces.h>
#include "Bundler.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class MonolithicTransformer : public Transformer
    {
    public:
        MonolithicTransformer(const ConfigParameters & readerConfig, size_t elementSize);

        virtual void SetEpochConfiguration(const EpochConfiguration& config);
        virtual std::vector<InputDescriptionPtr> getInputs() const override;
        virtual std::map<InputId, Sequence> getNextSequence() override;


        virtual std::map<InputId, Sequence> getNextSequence_new();


        virtual ~MonolithicTransformer()
        {}

    private:
        enum InputOutputTypes
        {
            real,
            category,
        };

        std::vector<FrameDescription> m_featureFrameDescriptions;
        std::vector<FrameDescription> m_labelFrameDescriptions;
        std::vector<InputDescriptionPtr> m_inputs;

        /*not used by necessary to initialize the source*/
        msra::asr::simplesenonehmm m_hset;
        unique_ptr<msra::dbn::latticesource> m_lattices;
        map<wstring, msra::lattices::lattice::htkmlfwordsequence> m_latticeMap;

        size_t m_elementSize; // size of the element, should go away probably and be taken from the layout?
        MemoryProviderPtr m_memoryProvider;
        bool m_noData;
        std::vector<size_t> m_featDims;
        std::map<std::wstring, size_t> m_nameToTypeMap;
        std::map<std::wstring, size_t> m_featureNameToIdMap;
        std::map<std::wstring, size_t> m_featureNameToDimMap;
        std::vector<size_t> m_labelDims;
        std::map<std::wstring, size_t> m_labelNameToIdMap;
        std::map<std::wstring, size_t> m_labelNameToDimMap;
        int m_verbosity;
        bool m_partialMinibatch;
        unique_ptr<msra::dbn::minibatchiterator> m_mbiter;
        std::shared_ptr<msra::dbn::Bundler> m_frameSource;

        std::map<std::wstring, size_t> m_nameToId;

        TransformerPtr m_transformer;
    };

    typedef std::shared_ptr<MonolithicTransformer> MonolithicTransformerPtr;
}}}
