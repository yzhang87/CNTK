#pragma once

#include "InnerInterfaces.h"
#include "commandArgUtil.h"
#include <opencv2/core/mat.hpp>

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageDataDeserializer : public DataDeserializer
    {
    public:
        ImageDataDeserializer(const ConfigParameters& config);
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

        class LabelGenerator
        {
        public:
            virtual void* GetLabelDataFor(size_t classId) = 0;
            virtual ~LabelGenerator() {}
        };

        typedef std::shared_ptr<LabelGenerator> LabelGeneratorPtr;

        void CreateSequenceDescriptions(std::string mapPath, size_t labelDimension);

        std::vector<ImageSequenceDescription> m_imageSequences;
        TimelineP m_sequences;

        TensorShapePtr m_labelSampleLayout;

        LabelGeneratorPtr m_labelGenerator;

        cv::Mat m_currentImage;
        ElementType m_featureElementType;
    };
}}}
