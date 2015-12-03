//
// <copyright file="ScpDataDeserializer.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>

#pragma once

#include "Basics.h"
#include "BlockRandomizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ScpDataDeserializer
    {
    public:
        std::vector<msra::dbn::utterancedesc> Parse(const std::vector<std::wstring>& scpFiles);
    };
}}}
