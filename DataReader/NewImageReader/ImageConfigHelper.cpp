#include "stdafx.h"
#include "ImageConfigHelper.h"
#include <algorithm>

namespace Microsoft { namespace MSR { namespace CNTK {

    static bool AreEqual(const std::string& s1, const std::string& s2)
    {
        return std::equal(s1.begin(), s1.end(), s2.begin(), [](const char& a, const char& b) { return std::tolower(a) == std::tolower(b); });
    }

    ImageConfigHelper::ImageConfigHelper(const ConfigParameters& config)
    {
        // TODO alexeyk: does not work for BrainScript, since configs cannot be copied
        using SectionT = std::pair<std::string, ConfigParameters>;
        auto getter = [&](const std::string& paramName) -> SectionT
        {
            auto sect = std::find_if(config.begin(), config.end(),
                [&](const std::pair<std::string, ConfigValue>& p)
            {
                return ConfigParameters(p.second).ExistsCurrent(paramName);
            });

            if (sect == config.end())
            {
                RuntimeError("ImageReader requires %s parameter.", paramName.c_str());
            }
            return{ (*sect).first, ConfigParameters((*sect).second) };
        };

        std::string rand = config(L"randomize", "auto");
        if (!AreEqual(rand, "auto"))
        {
            RuntimeError("Only Auto is currently supported.");
        }

        // REVIEW alexeyk: currently support only one feature and label section.
        SectionT featSect{ getter("width") };

        size_t w = featSect.second("width");
        size_t h = featSect.second("height");
        size_t c = featSect.second("channels");

        auto features = std::make_shared<InputDescription>();
        features->id = 0;
        features->type = it_feature;
        features->name = msra::strfun::utf16(featSect.first);
        features->sampleLayout = std::make_shared<ImageLayout>(ImageLayoutWHC(w, h, c));
        m_inputs.push_back(features);

        SectionT labSect{ getter("labelDim") };
        size_t labelDimension = labSect.second("labelDim");

        auto labels = std::make_shared<InputDescription>();
        labels->id = 1;
        labels->type = it_label;
        labels->name = msra::strfun::utf16(labSect.first);
        labels->sampleLayout = std::make_shared<ImageLayout>(ImageLayoutVector(labelDimension));
        m_inputs.push_back(labels);

        m_mapPath = config(L"file");
    }

    std::vector<InputDescriptionPtr> ImageConfigHelper::GetInputs() const
    {
        return m_inputs;
    }

    size_t ImageConfigHelper::GetFeatureInputIndex() const
    {
        // Hard-wired.
        return 0;
    }

    size_t ImageConfigHelper::GetLabelInputIndex() const
    {
        // Hard-wired.
        return 1;
    }

    std::string ImageConfigHelper::GetMapPath() const
    {
        return m_mapPath;
    }
}}}
