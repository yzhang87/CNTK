//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "ReaderInterfaces.h"
#include "ImageTransformers.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageReaderNew : public Reader
    {
    public:
        ImageReaderNew(const ConfigParameters& parameters,
            size_t elementSize,
            TransformerPtr transformer);

        std::vector<InputDescriptionPtr> GetInputs() override;
        EpochPtr StartNextEpoch(const EpochConfiguration& config) override;

    private:
        class EpochImplementation : public Epoch
        {
            ImageReaderNew* m_parent;

        public:
            EpochImplementation(ImageReaderNew* parent);
            virtual Minibatch ReadMinibatch() override;
            virtual ~EpochImplementation();
        };

        void InitFromConfig(const ConfigParameters& config);
        Minibatch GetMinibatch();

        // TODO: should be injected?
        TransformerPtr m_transformer;

        unsigned int m_seed;
        std::mt19937 m_rng;

        std::wstring m_featName;
        std::wstring m_labName;

        size_t m_featDim;
        size_t m_labDim;

        using StrIntPairT = std::pair<std::string, int>;
        std::vector<StrIntPairT> files;

        size_t m_epochSize;
        size_t m_mbSize;
        size_t m_epoch;

        size_t m_epochStart;
        size_t m_mbStart;

        //std::vector<ElemType> m_featBuf;
        //std::vector<ElemType> m_labBuf;

        bool m_imgListRand;

        MBLayoutPtr m_pMBLayout;

        size_t m_elementSize;
    };

}}}
