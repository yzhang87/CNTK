#pragma once

#include "ISource.h"

#include "DataReader.h"
#include "commandArgUtil.h" // for intargvector
#include "CUDAPageLockedMemAllocator.h"
#include "minibatchiterator.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType> // ?? Not clear, it should not be templatized.
    class HTKMLFSource : public ISource
    {
    public:
        HTKMLFSource(const ConfigParameters& readerConfig/*IBlockReader[] readers, framemode, inputs, ... config*/);
        virtual ~HTKMLFSource() {}

        virtual Timeline& getTimeline() override
        {
            throw std::logic_error("The method or operation is not implemented.");
        }

        virtual std::map<std::string, std::vector<sequence>> getSequenceById(std::vector<sequenceId> ids) override
        {
            throw std::logic_error("The method or operation is not implemented.");
        }

    private:
        std::map<std::wstring, size_t> m_nameToTypeMap;

        // Feature description, potentially shared with other layers
        std::vector<size_t> m_featDims;
        std::map<std::wstring, size_t> m_featureNameToIdMap;
        std::map<std::wstring, size_t> m_featureNameToDimMap;
        std::vector<std::shared_ptr<ElemType>> m_featuresBufferMultiIO;
        std::vector<size_t> m_featuresBufferAllocatedMultiIO;


        // Label description
        std::vector<size_t> m_labelDims;
        std::map<std::wstring, size_t> m_labelNameToIdMap;
        std::map<std::wstring, size_t> m_labelNameToDimMap;
        std::vector<std::shared_ptr<ElemType>> m_labelsBufferMultiIO;
        std::vector<size_t> m_labelsBufferAllocatedMultiIO;

        // Label target files???
        std::vector <bool> m_convertLabelsToTargetsMultiIO;
        std::vector<std::vector<std::vector<ElemType>>>m_labelToTargetMapMultiIO;

        // Loading from mmf files.
        msra::asr::simplesenonehmm m_hset;

        bool m_frameMode;
        int m_verbosity;
        bool m_partialMinibatch;

        // lattices
        unique_ptr<msra::dbn::latticesource> m_lattices;
        map<wstring, msra::lattices::lattice::htkmlfwordsequence> m_latticeMap;
        unique_ptr<msra::dbn::minibatchsource> m_frameSource;

        enum InputOutputTypes
        {
            real,
            category,
        };

    private:
        void HTKMLFSource<ElemType>::GetDataNamesFromConfig(
            const ConfigParameters& readerConfig,
            std::vector<std::wstring>& features,
            std::vector<std::wstring>& labels,
            std::vector<std::wstring>& hmms,
            std::vector<std::wstring>& lattices);

        size_t ReadLabelToTargetMappingFile(const std::wstring& labelToTargetMappingFile, const std::wstring& labelListFile, std::vector<std::vector<ElemType>>& labelToTargetMap);
    };
}}}