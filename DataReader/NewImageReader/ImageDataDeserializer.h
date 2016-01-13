//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "commandArgUtil.h"
#include <opencv2/core/mat.hpp>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Image data deserializer based on the OpenCV library.
    // The deserializer currently supports two output streams only: a feature and a label stream.
    // All sequences consist only of a single sample (image/label).
    // For features it uses dense storage format with different layour (dimensions) between different sequences.
    // For labels it uses sparse storage format.
    class ImageDataDeserializer : public DataDeserializer
    {
    public:
        explicit ImageDataDeserializer(const ConfigParameters& config);
        virtual ~ImageDataDeserializer() {}

        // Description of streams that this data deserializer provides.
        std::vector<StreamDescriptionPtr> GetStreams() const override;

        // Sets configuration for the current epoch.
        void SetEpochConfiguration(const EpochConfiguration& config) override;

        // Gets descriptions of all sequences the deserializer can produce.
        const Timeline& GetSequenceDescriptions() const override;

        // Get sequences by specified ids. Order of returned sequences correponds to the order of provided ids.
        std::vector<std::vector<SequenceDataPtr>> GetSequencesById(const std::vector<size_t>& ids) override;

        // Is be called by the randomizer for prefetching the next chunk.
        bool RequireChunk(size_t chunkIndex) override;

        // Is be called by the randomizer for releasing a prefetched chunk.
        void ReleaseChunk(size_t chunkIndex) override;

    private:
        // Creates a set of sequence descriptions.
        void CreateSequenceDescriptions(std::string mapPath, size_t labelDimension);

        // Image sequence descriptions. Currently, a sequence contains a single sample only.
        struct ImageSequenceDescription : public SequenceDescription
        {
            std::string path;
            size_t classId;
        };

        // A helper class for generation of type specific labels (currently float/double only).
        class LabelGenerator;
        typedef std::shared_ptr<LabelGenerator> LabelGeneratorPtr;
        LabelGeneratorPtr m_labelGenerator;

        // Sequence descriptions of all data.
        std::vector<ImageSequenceDescription> m_imageSequences;
        // Alias of m_imageSequences, providing pointers to the generic SequenceDescription.
        Timeline m_sequences;

        // Buffer to store label data.
        std::vector<SparseSequenceDataPtr> m_labels;
        // Buffer to store feature data.
        std::vector<cv::Mat> m_currentImages;

        // Exposed streams.
        std::vector<StreamDescriptionPtr> m_streams;

        // Element type of the feature/label stream (currently float/double only).
        ElementType m_featureElementType;
    };
}}}
