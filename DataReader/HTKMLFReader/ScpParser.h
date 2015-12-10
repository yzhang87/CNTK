//
// <copyright file="ScpParser.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>

#pragma once

#include "Basics.h"
#include "Bundler.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ScpParser
    {
    public:
        std::vector<msra::dbn::utterancedesc> Parse(const std::vector<std::wstring>& scpFiles);

        const std::map<std::wstring, size_t>& GetIdMap()
        {
            throw std::logic_error("Not implemented");
        }

        virtual const Timeline& GetTimeline() const
        {
            throw std::logic_error("Not implemented");
        }
    };

    typedef std::shared_ptr<ScpParser> ScpParserPtr;
}}}
