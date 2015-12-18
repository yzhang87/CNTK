//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "Basics.h"
#include "ReaderInterfaces.h"
#include "commandArgUtil.h"


namespace Microsoft { namespace MSR { namespace CNTK {

class ImageReaderNew : public Reader
{
public:
    std::vector<InputDescriptionPtr> GetInputs() override;
    EpochPtr StartNextEpoch(const EpochConfiguration& config) override;
};

typedef std::shared_ptr<ImageReaderNew> ImageReaderNewPtr;

}}}
