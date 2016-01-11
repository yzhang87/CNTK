#pragma once

#include "commandArgUtil.h"
#include <opencv2/core/mat.hpp>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageDataDeserializer : public DataDeserializer
    {
    public:
        ImageDataDeserializer(const ConfigParameters& config);
        virtual ~ImageDataDeserializer() {}

        std::vector<InputDescriptionPtr> GetInputs() const override;
        void SetEpochConfiguration(const EpochConfiguration& config) override;
        const Timeline& GetSequenceDescriptions() const override;
        std::vector<std::vector<SequenceData>> GetSequencesById(const std::vector<size_t> & ids) override;
        bool RequireChunk(size_t chunkIndex) override;
        void ReleaseChunk(size_t chunkIndex) override;

    private:
        struct ImageSequenceDescription : public SequenceDescription
        {
            std::string path;
            size_t classId;
        };

        class LabelGenerator
        {
        public:
            virtual void* GetLabelDataFor(size_t classId) = 0;
            virtual ~LabelGenerator() {}
        };

        typedef std::shared_ptr<LabelGenerator> LabelGeneratorPtr;

        void CreateSequenceDescriptions(std::string mapPath, size_t labelDimension);

        std::vector<ImageSequenceDescription> m_imageSequences;
        Timeline m_sequences;

        TensorShapePtr m_labelSampleLayout;

        LabelGeneratorPtr m_labelGenerator;

        std::vector<cv::Mat> m_currentImages;
        ElementType m_featureElementType;
    };
}}}
