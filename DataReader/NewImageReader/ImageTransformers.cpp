#include "stdafx.h"
#include "ImageTransformers.h"

#include "commandArgUtil.h"
#include "ConcStack.h"
#include <algorithm>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <random>

namespace Microsoft { namespace MSR { namespace CNTK {

    static bool AreEqual(const std::string& s1, const std::string& s2)
    {
        return std::equal(s1.begin(), s1.end(), s2.begin(), [](const char& a, const char& b) { return std::tolower(a) == std::tolower(b); });
    };

    CropTransformNew::CropTransformNew(
        TransformerPtr next,
        const std::set<std::wstring>& appliedStreams,
        const ConfigParameters& parameters,
        unsigned int seed)
        : m_seed(seed)
        , m_appliedStreams(appliedStreams)
        , m_next(next)
    {
        InitFromConfig(parameters);
    }

    void CropTransformNew::InitFromConfig(const ConfigParameters & config)
    {
        m_cropType = ParseCropType(config(L"cropType", ""));

        floatargvector cropRatio = config(L"cropRatio", "1.0");
        m_cropRatioMin = cropRatio[0];
        m_cropRatioMax = cropRatio[1];

        if (!(0 < m_cropRatioMin && m_cropRatioMin <= 1.0) ||
            !(0 < m_cropRatioMax && m_cropRatioMax <= 1.0) ||
            m_cropRatioMin > m_cropRatioMax)
        {
            RuntimeError("Invalid cropRatio value, must be > 0 and <= 1. cropMin must <= cropMax");
        }

        m_jitterType = ParseJitterType(config(L"jitterType", ""));

        if (!config.ExistsCurrent(L"hflip"))
        {
            m_hFlip = m_cropType == CropType::Random;
        }
        else
        {
            m_hFlip = std::stoi(config(L"hflip")) != 0;
        }
    }

    void CropTransformNew::SetEpochConfiguration(const EpochConfiguration& /*config*/)
    {
        const auto& inputs = m_next->GetInputs();
        m_appliedStreamsHash.resize(inputs.size(), false);

        for (const auto& input : inputs)
        {
            if (m_appliedStreams.find(input->name) != m_appliedStreams.end())
            {
                m_appliedStreamsHash[input->id] = true;
            }
        }

        // todo: check that all streams are exhausted.
    }

    std::vector<InputDescriptionPtr> CropTransformNew::GetInputs() const
    {
        return m_next->GetInputs();
    }

    SequenceData CropTransformNew::GetNextSequence()
    {
        SequenceData sample = m_next->GetNextSequence();
        if (sample.m_endOfEpoch)
        {
            return sample;
        }

        for (int i = 0; i < m_appliedStreamsHash.size(); ++i)
        {
            if (m_appliedStreamsHash[i])
            {
                sample.m_data[i] = Apply(sample.m_data[i]);
            }
        }

        throw std::logic_error("The method or operation is not implemented.");
    }

    Sequence CropTransformNew::Apply(Sequence& s)
    {
        int rows = static_cast<int>(s.layout->dimensions->GetHeight());
        int columns = static_cast<int>(s.layout->dimensions->GetHeight());
        int channels = static_cast<int>(s.layout->dimensions->GetNumChannels());

        int typeId = 0;
        if (s.layout->elementType == 8)
        {
            typeId = CV_64F;
        }
        else
        {
            typeId = CV_32F;
        }

        int type = CV_MAKETYPE(typeId, channels);
        cv::Mat mat(rows, columns, type, s.data);
        this->Apply(mat);

        Sequence result;
        result.layout = s.layout;
        result.numberOfSamples = result.numberOfSamples;
        result.data = mat.ptr();
        return result;
    }

    void CropTransformNew::Apply(cv::Mat& mat)
    {
        auto seed = m_seed;
        auto rng = m_rngs.pop_or_create([seed]() { return std::make_unique<std::mt19937>(seed); });

        double ratio = 1;
        switch (m_jitterType)
        {
        case RatioJitterType::None:
            ratio = m_cropRatioMin;
            break;
        case RatioJitterType::UniRatio:
            if (m_cropRatioMin == m_cropRatioMax)
            {
                ratio = m_cropRatioMin;
            }
            else
            {
                ratio = UniRealT(m_cropRatioMin, m_cropRatioMax)(*rng);
                assert(m_cropRatioMin <= ratio && ratio < m_cropRatioMax);
            }
            break;
        default:
            RuntimeError("Jitter type currently not implemented.");
        }

        mat = mat(GetCropRect(m_cropType, mat.rows, mat.cols, ratio, *rng));
        if (m_hFlip && std::bernoulli_distribution()(*rng))
            cv::flip(mat, mat, 1);

        m_rngs.push(std::move(rng));
    }

    CropTransformNew::CropType CropTransformNew::ParseCropType(const std::string& src)
    {
        if (src.empty() || AreEqual(src, "center"))
        {
            return CropType::Center;
        }

        if (AreEqual(src, "random"))
        {
            return CropType::Random;
        }

        RuntimeError("Invalid crop type: %s.", src.c_str());
    }

    CropTransformNew::RatioJitterType CropTransformNew::ParseJitterType(const std::string& src)
    {
        if (src.empty() || AreEqual(src, "none"))
        {
            return RatioJitterType::None;
        }

        if (AreEqual(src, "uniratio"))
        {
            return RatioJitterType::UniRatio;
        }

        if (AreEqual(src, "unilength"))
        {
            return RatioJitterType::UniLength;
        }

        if (AreEqual(src, "uniarea"))
        {
            return RatioJitterType::UniArea;
        }

        RuntimeError("Invalid jitter type: %s.", src.c_str());
    }

    cv::Rect CropTransformNew::GetCropRect(CropType type, int crow, int ccol, double cropRatio, std::mt19937& rng)
    {
        assert(crow > 0);
        assert(ccol > 0);
        assert(0 < cropRatio && cropRatio <= 1.0);

        int cropSize = static_cast<int>(std::min(crow, ccol) * cropRatio);
        int xOff = -1;
        int yOff = -1;
        switch (type)
        {
        case CropType::Center:
            xOff = (ccol - cropSize) / 2;
            yOff = (crow - cropSize) / 2;
            break;
        case CropType::Random:
            xOff = UniIntT(0, ccol - cropSize)(rng);
            yOff = UniIntT(0, crow - cropSize)(rng);
            break;
        default:
            assert(false);
        }

        assert(0 <= xOff && xOff <= ccol - cropSize);
        assert(0 <= yOff && yOff <= crow - cropSize);
        return cv::Rect(xOff, yOff, cropSize, cropSize);
    }
}}}