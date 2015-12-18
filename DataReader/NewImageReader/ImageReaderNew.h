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
        ImageReaderNew(const ConfigParameters& parameters, TransformerPtr transformer);

        std::vector<InputDescriptionPtr> GetInputs() override;
        EpochPtr StartNextEpoch(const EpochConfiguration& config) override;

    private:
        TransformerPtr m_transformer;
    };

}}}
