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
            size_t elementSize);

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

        TransformerPtr m_transformer;
        unsigned int m_seed;

        size_t m_featDim;
        size_t m_labDim;

        size_t m_mbSize;
        std::vector<char> m_featBuf;
        std::vector<char> m_labBuf;

        bool m_imgListRand;
        MBLayoutPtr m_pMBLayout;
        size_t m_elementSize;
        bool m_endOfEpoch;
    };

}}}
