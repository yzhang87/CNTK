#pragma once

#include <string>
#include <vector>
#include "commandArgUtil.h"
#include "InnerInterfaces.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageConfigHelper
    {
    public:
        ImageConfigHelper(const ConfigParameters& config);
        std::vector<InputDescriptionPtr> GetInputs() const;

        size_t GetFeatureInputIndex() const;
        size_t GetLabelInputIndex() const;
        std::string GetMapPath() const;

    private:
        std::string m_mapPath;
        std::vector<InputDescriptionPtr> m_inputs;
    };

    typedef std::shared_ptr<ImageConfigHelper> ImageConfigHelperPtr;
}}}
