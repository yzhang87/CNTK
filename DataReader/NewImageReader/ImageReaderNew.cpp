//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "ImageReaderNew.h"
#include "commandArgUtil.h"
#include "ImageTransformers.h"
#include "BlockRandomizer.h"
#include "ImageDataDeserializer.h"
#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    static bool AreEqual(const std::string& s1, const std::string& s2)
    {
        return std::equal(s1.begin(), s1.end(), s2.begin(), [](const char& a, const char& b) { return std::tolower(a) == std::tolower(b); });
    }

    ImageReaderNew::EpochImplementation::EpochImplementation(ImageReaderNew* parent)
        : m_parent(parent)
    {}

    ImageReaderNew::EpochImplementation::~EpochImplementation()
    {}

    Minibatch ImageReaderNew::EpochImplementation::ReadMinibatch()
    {
        return m_parent->GetMinibatch();
    }

    ImageReaderNew::ImageReaderNew(
        const ConfigParameters& parameters,
        size_t elementSize,
        TransformerPtr /*transformer*/) : m_elementSize(elementSize)
    {
        InitFromConfig(parameters);
    }

    void ImageReaderNew::InitFromConfig(const ConfigParameters& config)
    {
        std::string mapPath = config(L"file");
        std::ifstream mapFile(mapPath);
        if (!mapFile)
        {
            RuntimeError("Could not open %s for reading.", mapPath.c_str());
        }

        std::string line{ "" };
        for (size_t cline = 0; std::getline(mapFile, line); cline++)
        {
            std::stringstream ss{ line };
            std::string imgPath;
            std::string clsId;
            if (!std::getline(ss, imgPath, '\t') || !std::getline(ss, clsId, '\t'))
                RuntimeError("Invalid map file format, must contain 2 tab-delimited columns: %s, line: %d.", mapPath.c_str(), cline);
            files.push_back({ imgPath, std::stoi(clsId) });
        }

        DataDeserializerPtr deserializer = std::make_shared<ImageDataDeserializer>(config, m_elementSize);
        TransformerPtr randomizer = std::make_shared<BlockRandomizer>(0, SIZE_MAX, deserializer);
        TransformerPtr cropper = std::make_shared<CropTransformNew>(randomizer, std::set<std::wstring> { L"features" }, config, m_seed);
        TransformerPtr scaler = std::make_shared<ScaleTransform>(cropper, std::set<std::wstring> { L"features" }, sizeof(m_elementSize) == 4 ? CV_32F : CV_64F, m_seed, config);
        TransformerPtr mean = std::make_shared<MeanTransform>(scaler, std::set<std::wstring> { L"features" });
        m_transformer = mean;

        using SectionT = std::pair<std::string, ConfigParameters>;      // TODO: does not work for BrainScript, since configs cannot be copied
        auto gettter = [&](const std::string& paramName) -> SectionT
        {
            auto sect = std::find_if(config.begin(), config.end(),
                [&](const std::pair<std::string, ConfigValue>& p) { return ConfigParameters(p.second).ExistsCurrent(paramName); });
            if (sect == config.end())
                RuntimeError("ImageReader requires %s parameter.", paramName.c_str());
            return{ (*sect).first, ConfigParameters((*sect).second) };
        };

        // REVIEW alexeyk: currently support only one feature and label section.
        SectionT featSect{ gettter("width") };
        m_featName = msra::strfun::utf16(featSect.first);

        // REVIEW alexeyk: w, h and c will be read again in ScaleTransformOld.
        size_t w = featSect.second("width");
        size_t h = featSect.second("height");
        size_t c = featSect.second("channels");
        m_featDim = w * h * c;

        SectionT labSect{ gettter("labelDim") };
        m_labName = msra::strfun::utf16(labSect.first);
        m_labDim = labSect.second("labelDim");

        std::string rand = config(L"randomize", "auto");
        if (!AreEqual(rand, "auto"))
        {
            RuntimeError("Only Auto is currently supported.");
        }

        m_epochStart = 0;
        m_mbStart = 0;
    }


    std::vector<InputDescriptionPtr> ImageReaderNew::GetInputs()
    {
        return m_transformer->GetInputs();
    }

    EpochPtr ImageReaderNew::StartNextEpoch(const EpochConfiguration& config)
    {
        assert(config.minibatchSize > 0);
        assert(config.totalSize > 0);

        m_epochSize = (config.totalSize == requestDataSize ? files.size() : config.totalSize);
        m_mbSize = config.minibatchSize;

        // REVIEW alexeyk: if user provides epoch size explicitly then we assume epoch size is a multiple of mbsize, is this ok?
        assert(config.totalSize == requestDataSize || (m_epochSize % m_mbSize) == 0);

        m_epoch = config.index;
        m_epochStart = m_epoch * m_epochSize;
        if (m_epochStart >= files.size())
        {
            m_epochStart = 0;
            m_mbStart = 0;
        }

        //m_featBuf.resize(m_mbSize * m_featDim);
        //m_labBuf.resize(m_mbSize * m_labDim);

        return std::make_shared<EpochImplementation>(this);
    }

    Minibatch ImageReaderNew::GetMinibatch()
    {
//        assert(matrices.size() > 0);
//        assert(matrices.find(m_featName) != matrices.end());
//        assert(m_mbSize > 0);
//
//        if (m_mbStart >= files.size() || m_mbStart >= m_epochStart + m_epochSize)
//            return false;
//
//        size_t mbLim = m_mbStart + m_mbSize;
//        if (mbLim > files.size())
//            mbLim = files.size();
//
//        std::fill(m_labBuf.begin(), m_labBuf.end(), static_cast<ElemType>(0));
//
//#pragma omp parallel for ordered schedule(dynamic)
//        for (long long i = 0; i < static_cast<long long>(mbLim - m_mbStart); i++)
//        {
//            const auto& p = files[i + m_mbStart];
//            cv::Mat img{ cv::imread(p.first, cv::IMREAD_COLOR) };
//            for (auto& t : m_transforms)
//                t->Apply(img);
//
//            assert(img.isContinuous());
//            auto data = reinterpret_cast<ElemType*>(img.ptr());
//            std::copy(data, data + m_featDim, m_featBuf.begin() + m_featDim * i);
//            m_labBuf[m_labDim * i + p.second] = 1;
//        }
//
//        size_t mbSize = mbLim - m_mbStart;
//        features.SetValue(m_featDim, mbSize, features.GetDeviceId(), m_featBuf.data(), matrixFlagNormal);
//        labels.SetValue(m_labDim, mbSize, labels.GetDeviceId(), m_labBuf.data(), matrixFlagNormal);
//        m_pMBLayout->Init(mbSize, 1, false);
//
//        m_mbStart = mbLim;
//        return true;
        throw std::runtime_error("Not implemented.");
    }
}}}

