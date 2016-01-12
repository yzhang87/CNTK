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

        std::vector<StreamDescriptionPtr> GetStreams() const override;
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
            virtual void ReadLabelDataFor(SequenceData& data, size_t classId) = 0;
            virtual ~LabelGenerator() {}
        };

        typedef std::shared_ptr<LabelGenerator> LabelGeneratorPtr;

        void CreateSequenceDescriptions(std::string mapPath, size_t labelDimension);

        std::vector<ImageSequenceDescription> m_imageSequences;
        std::vector<SequenceData> m_labels;
        LabelGeneratorPtr m_labelGenerator;
        std::vector<cv::Mat> m_currentImages;
        std::vector<StreamDescriptionPtr> m_streams;

        Timeline m_sequences;
        ElementType m_featureElementType;
    };
}}}
