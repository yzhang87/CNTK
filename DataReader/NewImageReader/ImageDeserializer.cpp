#include "stdafx.h"
#include "ImageDeserializer.h"
#include <opencv2/opencv.hpp>

namespace Microsoft { namespace MSR { namespace CNTK {

    ImageDataDeserializer::ImageDataDeserializer(const ConfigParameters& config, size_t elementSize)
        : m_elementSize(elementSize)
    {
        using SectionT = std::pair<std::string, ConfigParameters>;      // TODO: does not work for BrainScript, since configs cannot be copied
        auto getter = [&](const std::string& paramName) -> SectionT
        {
            auto sect = std::find_if(config.begin(), config.end(),
                [&](const std::pair<std::string, ConfigValue>& p)
                {
                    return ConfigParameters(p.second).ExistsCurrent(paramName);
                });
            if (sect == config.end())
                RuntimeError("ImageReader requires %s parameter.", paramName.c_str());
            return{ (*sect).first, ConfigParameters((*sect).second) };
        };

        // REVIEW alexeyk: currently support only one feature and label section.
        SectionT featSect{ getter("width") };
        m_featName = msra::strfun::utf16(featSect.first);
        // REVIEW alexeyk: w, h and c will be read again in ScaleTransform.
        size_t w = featSect.second("width");
        size_t h = featSect.second("height");
        size_t c = featSect.second("channels");
        m_featDim = w * h * c;
        // TODO we should not need this?

        SectionT labSect{ getter("labelDim") };
        m_labName = msra::strfun::utf16(labSect.first);
        m_labDim = labSect.second("labelDim");

        std::string mapPath = config(L"file");
        std::ifstream mapFile(mapPath);
        if (!mapFile)
            RuntimeError("Could not open %s for reading.", mapPath.c_str());

        std::string line{ "" };

        ImageSequenceDescription description;

        description.numberOfSamples = 1;
        description.isValid = true;

        for (size_t cline = 0; std::getline(mapFile, line); cline++)
        {
            std::stringstream ss{ line };
            std::string imgPath;
            std::string clsId;
            if (!std::getline(ss, imgPath, '\t') || !std::getline(ss, clsId, '\t'))
                RuntimeError("Invalid map file format, must contain 2 tab-delimited columns: %s, line: %d.", mapPath.c_str(), cline);
            description.id = cline;
            // Put each image in its own chunk
            description.chunkId = cline;
            description.path = imgPath;
            description.classId = std::stoi(clsId);
            assert(description.classId < m_labDim);
            m_imageSequences.push_back(description);
            m_sequences.push_back(&m_imageSequences[cline]);
        }
    }

    std::vector<InputDescriptionPtr> ImageDataDeserializer::GetInputs() const
    {
        std::vector<InputDescriptionPtr> dummy;
        return dummy;
    }

    void ImageDataDeserializer::SetEpochConfiguration(const EpochConfiguration& /* config */)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    const TimelineP& ImageDataDeserializer::GetSequenceDescriptions() const
    {
        return m_sequences;
    }

    std::vector<Sequence> ImageDataDeserializer::GetSequenceById(size_t id)
    {
        assert(id < m_imageSequences.size());
        const auto & imageSequence = m_imageSequences[id];

        // Construct image
        Sequence image;

        cv::Mat cvImage{ cv::imread(imageSequence.path, cv::IMREAD_COLOR) };
        assert(cvImage.isContinuous());
        image.data = cvImage.ptr();
        // TODO label.layout !!

        // Construct label
        Sequence label;
        // TODO label.layout !!
        if (m_elementSize == sizeof(float))
        {
            float* tmp = new float[m_labDim];
            memset(tmp, 0, m_elementSize * m_labDim);
            tmp[imageSequence.classId] = 1;
            label.data = tmp;
        }
        else
        {
            double* tmp = new double[m_labDim];
            memset(tmp, 0, m_elementSize * m_labDim);
            tmp[imageSequence.classId] = 1;
            label.data = tmp;
        }

        label.numberOfSamples = m_sequences[id]->numberOfSamples;

        return std::vector<Sequence> { image, label };
    }

    bool ImageDataDeserializer::RequireChunk(size_t /* chunkIndex */)
    {
        return true;
    }

    void ImageDataDeserializer::ReleaseChunk(size_t /* chunkIndex */)
    {
    }

}}}