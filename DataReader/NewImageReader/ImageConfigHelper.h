//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include <string>
#include <vector>
#include "commandArgUtil.h"
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageConfigHelper
    {
    public:
        ImageConfigHelper(const ConfigParameters& config);
        std::vector<StreamDescriptionPtr> GetStreams() const;

        size_t GetFeatureStreamId() const;
        size_t GetLabelStreamId() const;
        std::string GetMapPath() const;

    private:
        std::string m_mapPath;
        std::vector<StreamDescriptionPtr> m_streams;
    };

    typedef std::shared_ptr<ImageConfigHelper> ImageConfigHelperPtr;
}}}
