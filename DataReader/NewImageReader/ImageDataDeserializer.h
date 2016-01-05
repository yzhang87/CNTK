#pragma once

#include "InnerInterfaces.h"
#include "commandArgUtil.h"
#include <opencv2/core/mat.hpp>

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageDataDeserializer : public DataDeserializer
    {
    public:
        ImageDataDeserializer(const ConfigParameters& config, size_t elementSize);
        virtual ~ImageDataDeserializer() {}

        std::vector<InputDescriptionPtr> GetInputs() const override;
        void SetEpochConfiguration(const EpochConfiguration& config) override;
        const TimelineP& GetSequenceDescriptions() const override;
        std::vector<Sequence> GetSequenceById(size_t id) override;
        bool RequireChunk(size_t chunkIndex) override;
        void ReleaseChunk(size_t chunkIndex) override;

    private:
        struct ImageSequenceDescription : public SequenceDescription
        {
            std::string path;
            size_t classId;
        };

        void CreateSequenceDescriptions(const ConfigParameters& config, size_t labelDimension);

        std::vector<ImageSequenceDescription> m_imageSequences;
        TimelineP m_sequences;

        std::vector<InputDescriptionPtr> m_inputs;
        std::vector<float> m_floatLabelData;
        std::vector<double> m_doubleLabelData;
        cv::Mat m_currentImage;

        size_t m_elementSize;
        int m_imgChannels;
    };
}}}