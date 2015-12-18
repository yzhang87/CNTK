//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "ImageReaderNew.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    std::vector<InputDescriptionPtr> ImageReaderNew::GetInputs()
    {
        std::vector<InputDescriptionPtr> dummy;
        return dummy;
    }

    EpochPtr ImageReaderNew::StartNextEpoch(const EpochConfiguration& /* config */)
    {
        return nullptr;
    }

}}}

