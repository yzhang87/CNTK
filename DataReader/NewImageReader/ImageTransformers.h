#pragma once

#include <set>
#include <fstream>
#include <unordered_map>
#include <random>
#include <opencv2/opencv.hpp>

#include "InnerInterfaces.h"
#include "ConcStack.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ConfigParameters;

    class CropTransformNew : public Transformer
    {
    public:
        CropTransformNew(
            TransformerPtr next,
            const std::set<std::wstring>& appliedStreams,
            const ConfigParameters& parameters,
            unsigned int seed);

        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;

        virtual std::vector<InputDescriptionPtr> GetInputs() const override;

        virtual SequenceData GetNextSequence() override;

    private:
        using UniRealT = std::uniform_real_distribution<double>;
        using UniIntT = std::uniform_int_distribution<int>;

        enum class CropType { Center = 0, Random = 1 };
        enum class RatioJitterType
        {
            None = 0,
            UniRatio = 1,
            UniLength = 2,
            UniArea = 3
        };

        CropType ParseCropType(const std::string& src);
        RatioJitterType ParseJitterType(const std::string& src);
        cv::Rect GetCropRect(CropType type, int crow, int ccol, double cropRatio, std::mt19937& rng);
        void InitFromConfig(const ConfigParameters & config);

        Sequence Apply(Sequence& mat);
        void Apply(cv::Mat& mat);


    private:
        unsigned int m_seed;
        conc_stack<std::unique_ptr<std::mt19937>> m_rngs;

        CropType m_cropType;
        double m_cropRatioMin;
        double m_cropRatioMax;
        RatioJitterType m_jitterType;
        bool m_hFlip;

        std::set<std::wstring> m_appliedStreams;
        std::vector<bool> m_appliedStreamsHash;
        TransformerPtr m_next;
    };
}}}