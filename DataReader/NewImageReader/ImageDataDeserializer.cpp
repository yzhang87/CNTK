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

    ImageDataDeserializer::ImageDataDeserializer(const ConfigParameters& config)
    {
        auto configHelper = ImageConfigHelper(config);
        m_inputs = configHelper.GetInputs();
        assert(m_inputs.size() == 2);
        const auto & label = m_inputs[configHelper.GetLabelInputIndex()];
        const auto & feature = m_inputs[configHelper.GetFeatureInputIndex()];

        m_featureElementType = feature->elementType;
        m_labelSampleLayout = label->sampleLayout;
        size_t labelDimension = m_labelSampleLayout->GetHeight();

        if (label->elementType == ElementType::et_float)
        {
            m_labelGenerator = std::make_shared<TypedLabelGenerator<float>>(labelDimension);
        }
        else if (label->elementType == ElementType::et_double)
        {
            m_labelGenerator = std::make_shared<TypedLabelGenerator<double>>(labelDimension);
        }
        else
        {
            RuntimeError("Unsupported label element type %ull.", label->elementType);
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
        return m_inputs;
    }

    void ImageDataDeserializer::SetEpochConfiguration(const EpochConfiguration& /* config */)
    {
    }

    const Timeline& ImageDataDeserializer::GetSequenceDescriptions() const
    {
        return m_sequences;
    }

    std::vector<std::vector<SequenceData>> ImageDataDeserializer::GetSequencesById(const std::vector<size_t> & ids)
    {
        assert(0 < ids.size());

        std::vector<std::vector<SequenceData>> result;

        m_currentImages.resize(ids.size());
        result.resize(ids.size());

#pragma omp parallel for ordered schedule(dynamic)
        for (int i = 0; i < ids.size(); ++i)
        {
            assert(ids[i] < m_imageSequences.size());
            const auto& imageSequence = m_imageSequences[ids[i]];

            // Construct image
            SequenceData image;

            m_currentImages[i] = std::move(cv::imread(imageSequence.path, cv::IMREAD_COLOR));
            cv::Mat& cvImage = m_currentImages[i];
            assert(cvImage.isContinuous());

            // Convert element type.
            // TODO in original image reader, this conversion happened later. Should we support all native CV element types to be able to match this behavior?
            int dataType = m_featureElementType == ElementType::et_float ? CV_32F : CV_64F;
            if (cvImage.type() != CV_MAKETYPE(dataType, cvImage.channels()))
            {
                cvImage.convertTo(cvImage, dataType);
            }

            image.data = cvImage.ptr();
            image.layout = std::make_shared<ImageLayout>(ImageLayoutWHC(cvImage.cols, cvImage.rows, cvImage.channels()));;
            image.numberOfSamples = imageSequence.numberOfSamples;

            // Construct label
            SequenceData label;
            label.data = m_labelGenerator->GetLabelDataFor(imageSequence.classId);
            label.layout = m_labelSampleLayout;
            label.numberOfSamples = imageSequence.numberOfSamples;

            result[i] = std::move(std::vector<SequenceData> { image, label });
        }

        return result;
    }

    // TODO RequireChunk: re-ahead here (in cooperation with randomizer requesting images)
    bool ImageDataDeserializer::RequireChunk(size_t /* chunkIndex */)
    {
        return true;
    }

    void ImageDataDeserializer::ReleaseChunk(size_t /* chunkIndex */)
    {
    }
}}}
