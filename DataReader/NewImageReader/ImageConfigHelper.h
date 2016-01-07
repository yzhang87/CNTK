#pragma once

#include <utility>
#include <string>
#include <vector>
#include "commandArgUtil.h"
#include "InnerInterfaces.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageConfigHelper
    {
        std::string m_mapPath;
        std::vector<InputDescriptionPtr> m_inputs;

    public:
        ImageConfigHelper(const ConfigParameters& config);
        std::vector<InputDescriptionPtr> GetInputs() const;
        
        // TODO only single feature and label are supported
        size_t GetFeatureInputIndex() const;
        size_t GetLabelInputIndex() const;
        std::string GetMapPath() const;
    };

    typedef std::shared_ptr<ImageConfigHelper> ImageConfigHelperPtr;
}}}
