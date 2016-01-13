//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include <set>
#include <unordered_map>
#include <random>
#include <opencv2/opencv.hpp>

#include "Transformer.h"
#include "ConcStack.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ConfigParameters;

    class BaseTransformer : public Transformer
    {
    public:
        BaseTransformer();

        virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;
        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;
        virtual Sequences GetNextSequences(size_t count) override;
        virtual std::vector<StreamDescriptionPtr> GetStreams() const override;

    protected:
        using UniRealT = std::uniform_real_distribution<double>;
        using UniIntT = std::uniform_int_distribution<int>;

        SequenceDataPtr Apply(const DenseSequenceData& mat, StreamDescriptionPtr stream);
        virtual void Apply(cv::Mat& mat) = 0;
        unsigned int GetSeed() const
        {
            return m_seed;
        }

        const std::vector<StreamId> & GetFeatureStreamIds() const;
        std::vector<StreamDescriptionPtr> m_streams;

    private:
        std::vector<StreamId> m_featureStreamIds;
        TransformerPtr m_next;
        unsigned int m_seed;
        cv::Mat m_buffer;
    };

    class CropTransform : public BaseTransformer
    {
    public:
        CropTransform();
        virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;

    protected:
        virtual void Apply(cv::Mat& mat) override;

    private:
        enum class CropType { Center = 0, Random = 1 };
        enum class RatioJitterType
        {
            None = 0,
            UniRatio = 1,
            UniLength = 2,
            UniArea = 3
        };

        void InitFromConfig(const ConfigParameters& config);
        CropType ParseCropType(const std::string& src);
        RatioJitterType ParseJitterType(const std::string& src);
        cv::Rect GetCropRect(CropType type, int crow, int ccol, double cropRatio, std::mt19937& rng);

        conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
        CropType m_cropType;
        double m_cropRatioMin;
        double m_cropRatioMax;
        RatioJitterType m_jitterType;
        bool m_hFlip;
    };

    class ScaleTransform : public BaseTransformer
    {
    public:
        ScaleTransform();
        virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;

    private:
        void InitFromConfig(const ConfigParameters& config);
        virtual void Apply(cv::Mat& mat) override;

        using StrToIntMapT = std::unordered_map<std::string, int>;
        StrToIntMapT m_interpMap;
        std::vector<int> m_interp;

        conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
        int m_dataType;
        size_t m_imgWidth;
        size_t m_imgHeight;
        size_t m_imgChannels;
    };

    class MeanTransform : public BaseTransformer
    {
    public:
        MeanTransform();
        virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;

    private:
        virtual void Apply(cv::Mat& mat) override;
        void InitFromConfig(const ConfigParameters& config);

        cv::Mat m_meanImg;
    };
}}}
