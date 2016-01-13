#include "stdafx.h"
#include "ImageTransformers.h"

#include "commandArgUtil.h"
#include "ConcStack.h"
#include <algorithm>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <random>
#include "ImageConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    static bool AreEqual(const std::string& s1, const std::string& s2)
    {
        return std::equal(s1.begin(), s1.end(), s2.begin(), [](const char& a, const char& b) { return std::tolower(a) == std::tolower(b); });
    };

    BaseTransformer::BaseTransformer()
        : m_next(nullptr)
    {
    }

    void BaseTransformer::Initialize(TransformerPtr next, const ConfigParameters& readerConfig)
    {
        m_next = next;
        m_streams = GetStreams();
        m_seed = std::stoi(readerConfig(L"seed", "0"));

        ImageConfigHelper config(readerConfig);

        // Currently we only support a single stream.
        m_featureStreamIds.push_back(config.GetFeatureStreamId());
    }

    std::vector<StreamDescriptionPtr> BaseTransformer::GetStreams() const
    {
        return m_next->GetStreams();
    }

    void BaseTransformer::SetEpochConfiguration(const EpochConfiguration& config)
    {
        assert(m_next != nullptr);
        m_next->SetEpochConfiguration(config);
    }

    Sequences BaseTransformer::GetNextSequences(size_t count)
    {
        assert(m_next != nullptr);
        Sequences samples = m_next->GetNextSequences(count);

        if (samples.m_endOfEpoch)
        {
            return samples;
        }

        // TODO parallelization like in the original image reader
        for (auto & sample : samples.m_data)
        {
            assert(sample.size() == m_streams.size());

            for (auto id : m_featureStreamIds)
            {
                assert(m_streams[id]->storageType == StorageType::st_dense);
                const DenseSequenceData& sequence = reinterpret_cast<DenseSequenceData&>(*sample[id]);

                sample[id] = Apply(sequence, m_streams[id]);
            }
        }

        return samples;
    }

    SequenceDataPtr BaseTransformer::Apply(const DenseSequenceData& sequence, StreamDescriptionPtr stream)
    {
        int rows = static_cast<int>(sequence.sampleLayout->GetWidth());
        int columns = static_cast<int>(sequence.sampleLayout->GetHeight());
        int channels = static_cast<int>(sequence.sampleLayout->GetNumChannels());

        int typeId = 0;
        if (stream->elementType == ElementType::et_double)
        {
            typeId = CV_64F;
        }
        else if (stream->elementType == ElementType::et_float)
        {
            typeId = CV_32F;
        }
        else
        {
            RuntimeError("Unsupported type");
        }

        int type = CV_MAKETYPE(typeId, channels);
        m_buffer = cv::Mat(rows, columns, type, sequence.data);
        this->Apply(m_buffer);

        DenseSequenceDataPtr result = std::make_shared<DenseSequenceData>();
        result->sampleLayout = std::make_shared<ImageLayout>(
            ImageLayoutWHC(m_buffer.cols, m_buffer.rows, m_buffer.channels()));
        result->numberOfSamples = sequence.numberOfSamples;
        result->data = m_buffer.ptr();
        return result;
    }

    const std::vector<StreamId>& BaseTransformer::GetFeatureStreamIds() const
    {
        return m_featureStreamIds;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    CropTransform::CropTransform()
    {
    }

    void CropTransform::Initialize(TransformerPtr next, const ConfigParameters& readerConfig)
    {
        BaseTransformer::Initialize(next, readerConfig);
        auto featureStreamIds = GetFeatureStreamIds();

        if (featureStreamIds.size() != 1)
        {
            RuntimeError("Only a single feature stream is supported.");
        }

        InitFromConfig(readerConfig(m_streams[featureStreamIds[0]]->name));
    }

    void CropTransform::InitFromConfig(const ConfigParameters & config)
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

    void CropTransform::Apply(cv::Mat& mat)
    {
        auto seed = GetSeed();
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

    CropTransform::CropType CropTransform::ParseCropType(const std::string& src)
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

    CropTransform::RatioJitterType CropTransform::ParseJitterType(const std::string& src)
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

    cv::Rect CropTransform::GetCropRect(CropType type, int crow, int ccol, double cropRatio, std::mt19937& rng)
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

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ScaleTransform::ScaleTransform()
    {
    }

    void ScaleTransform::Initialize(TransformerPtr next, const ConfigParameters& readerConfig)
    {
        BaseTransformer::Initialize(next, readerConfig);
        m_interpMap.emplace("nearest", cv::INTER_NEAREST);
        m_interpMap.emplace("linear", cv::INTER_LINEAR);
        m_interpMap.emplace("cubic", cv::INTER_CUBIC);
        m_interpMap.emplace("lanczos", cv::INTER_LANCZOS4);

        auto featureStreamIds = GetFeatureStreamIds();

        if (featureStreamIds.size() != 1)
        {
            RuntimeError("Only a single feature stream is supported.");
        }

        const auto & feature = m_streams[featureStreamIds[0]];
        m_dataType = feature->elementType == ElementType::et_float ? CV_32F : CV_64F;

        InitFromConfig(readerConfig(feature->name));
    }

    void ScaleTransform::InitFromConfig(const ConfigParameters& config)
    {
        m_imgWidth = config(L"width");
        m_imgHeight = config(L"height");
        m_imgChannels = config(L"channels");

        size_t cfeat = m_imgWidth * m_imgHeight * m_imgChannels;
        if (cfeat == 0 || cfeat > std::numeric_limits<size_t>().max() / 2)
            RuntimeError("Invalid image dimensions.");

        m_interp.clear();
        std::stringstream ss{ config(L"interpolations", "") };
        for (std::string token = ""; std::getline(ss, token, ':');)
        {
            // Explicit cast required for GCC.
            std::transform(token.begin(), token.end(), token.begin(), (int(*)(int))std::tolower);
            StrToIntMapT::const_iterator res = m_interpMap.find(token);
            if (res != m_interpMap.end())
                m_interp.push_back((*res).second);
        }

        if (m_interp.size() == 0)
            m_interp.push_back(cv::INTER_LINEAR);
    }

    void ScaleTransform::Apply(cv::Mat& mat)
    {
        // If matrix has not been converted to the right type, do it now as rescaling requires floating point type.
        //
        if (mat.type() != CV_MAKETYPE(m_dataType, m_imgChannels))
            mat.convertTo(mat, m_dataType);

        auto seed = GetSeed();
        auto rng = m_rngs.pop_or_create([seed]() { return std::make_unique<std::mt19937>(seed); });

        assert(m_interp.size() > 0);
        cv::resize(mat, mat, cv::Size(static_cast<int>(m_imgWidth), static_cast<int>(m_imgHeight)), 0, 0,
            m_interp[UniIntT(0, static_cast<int>(m_interp.size()) - 1)(*rng)]);

        m_rngs.push(std::move(rng));
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    MeanTransform::MeanTransform()
    {
    }

    void MeanTransform::Initialize(TransformerPtr next, const ConfigParameters& readerConfig)
    {
        BaseTransformer::Initialize(next, readerConfig);
    }

    void MeanTransform::InitFromConfig(const ConfigParameters & config)
    {
        std::wstring meanFile = config(L"meanFile", L"");
        if (meanFile.empty())
            m_meanImg.release();
        else
        {
            cv::FileStorage fs;
            // REVIEW alexeyk: this sort of defeats the purpose of using wstring at all...  [fseide] no, only OpenCV has this problem.
            fs.open(msra::strfun::utf8(meanFile).c_str(), cv::FileStorage::READ);
            if (!fs.isOpened())
                RuntimeError("Could not open file: %ls", meanFile.c_str());
            fs["MeanImg"] >> m_meanImg;
            int cchan;
            fs["Channel"] >> cchan;
            int crow;
            fs["Row"] >> crow;
            int ccol;
            fs["Col"] >> ccol;
            if (cchan * crow * ccol != m_meanImg.channels() * m_meanImg.rows * m_meanImg.cols)
                RuntimeError("Invalid data in file: %ls", meanFile.c_str());
            fs.release();
            m_meanImg = m_meanImg.reshape(cchan, crow);
        }
    }

    void MeanTransform::Apply(cv::Mat& mat)
    {
        assert(m_meanImg.size() == cv::Size(0, 0) || (m_meanImg.size() == mat.size() && m_meanImg.channels() == mat.channels()));

        // REVIEW alexeyk: check type conversion (float/double).
        if (m_meanImg.size() == mat.size())
            mat = mat - m_meanImg;
    }
}}}
