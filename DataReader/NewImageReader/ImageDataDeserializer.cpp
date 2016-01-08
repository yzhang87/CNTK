#include "stdafx.h"
#include "ImageDataDeserializer.h"
#include "ImageConfigHelper.h"
#include <opencv2/opencv.hpp>

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class TElement>
    class TypedLabelGenerator : public ImageDataDeserializer::LabelGenerator
    {
    public:
        TypedLabelGenerator(size_t dimensions)
        {
            m_labelData.resize(dimensions, 0);
        }

        virtual void* GetLabelDataFor(size_t classId) override
        {
            std::fill(m_labelData.begin(), m_labelData.end(), static_cast<TElement>(0));
            m_labelData[classId] = 1;
            return &m_labelData[0];
        }

    private:
        std::vector<TElement> m_labelData;
    };

    ImageDataDeserializer::ImageDataDeserializer(const ConfigParameters& config, ElementType elementType)
        : m_elementType(elementType)
    {
        auto configHelper = ImageConfigHelper(config);
        auto inputs = configHelper.GetInputs();
        assert(inputs.size() == 2);
        const auto & labels = inputs[configHelper.GetLabelInputIndex()];

        m_labelSampleLayout = labels->sampleLayout;

        size_t labelDimension = m_labelSampleLayout->GetHeight();
        if (m_elementType == ElementType::et_float)
        {
            m_labelGenerator = std::make_shared<TypedLabelGenerator<float>>(labelDimension);
        }
        else if (m_elementType == ElementType::et_double)
        {
            m_labelGenerator = std::make_shared<TypedLabelGenerator<double>>(labelDimension);
        }
        else
        {
            RuntimeError("Unsupported element type %ull.", m_elementType);
        }

        CreateSequenceDescriptions(configHelper.GetMapPath(), labelDimension);
    }

    void ImageDataDeserializer::CreateSequenceDescriptions(std::string mapPath, size_t labelDimension)
    {
        UNREFERENCED_PARAMETER(labelDimension);

        std::ifstream mapFile(mapPath);
        if (!mapFile)
        {
            RuntimeError("Could not open %s for reading.", mapPath.c_str());
        }

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
            {
                RuntimeError("Invalid map file format, must contain 2 tab-delimited columns: %s, line: %d.", mapPath.c_str(), cline);
            }

            description.id = cline;
            description.chunkId = cline;
            description.path = imgPath;
            description.classId = std::stoi(clsId);
            assert(description.classId < labelDimension);
            m_imageSequences.push_back(description);
        }

        for (const auto& sequence : m_imageSequences)
        {
            m_sequences.push_back(&sequence);
        }
    }

    std::vector<InputDescriptionPtr> ImageDataDeserializer::GetInputs() const
    {
        assert(false);
        throw std::runtime_error("Not supported");
    }

    void ImageDataDeserializer::SetEpochConfiguration(const EpochConfiguration& /* config */)
    {
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

        m_currentImage = cv::imread(imageSequence.path, cv::IMREAD_COLOR);
        assert(m_currentImage.isContinuous());

        // Convert element type.
        int dataType = m_elementType == et_float ? CV_32F : CV_64F;
        if (m_currentImage.type() != CV_MAKETYPE(dataType, m_currentImage.channels()))
        {
            m_currentImage.convertTo(m_currentImage, dataType);
        }

        image.data = m_currentImage.ptr();
        image.layout = std::make_shared<ImageLayout>(ImageLayoutWHC(m_currentImage.cols, m_currentImage.rows, m_currentImage.channels()));;
        image.numberOfSamples = imageSequence.numberOfSamples;

        // Construct label
        Sequence label;
        label.data = m_labelGenerator->GetLabelDataFor(imageSequence.classId);
        label.layout = m_labelSampleLayout;
        label.numberOfSamples = imageSequence.numberOfSamples;
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
