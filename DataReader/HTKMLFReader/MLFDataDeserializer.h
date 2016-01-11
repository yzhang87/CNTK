//
// <copyright file="MLFDataDeserializer.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "DataDeserializer.h"
#include "HTKDataDeserializer.h"
#include "biggrowablevectors.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    struct MLFUtterance : public SequenceDescription
    {
        // Where the sequence is stored in m_classIds
        size_t sequenceStart;
    };

    struct MLFFrame : public SequenceDescription
    {
        // Where the sequence is stored in m_classIds
        size_t index;
    };

    class MLFDataDeserializer : public DataDeserializer
    {
        size_t m_dimension;
        TensorShapePtr m_layout;
        std::wstring m_stateListPath;
        std::vector<std::wstring> m_mlfPaths;
        const HTKDataDeserializer* m_featureDeserializer;

        // [classidsbegin+t] concatenation of all state sequences
        msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE> m_classIds;

        std::vector<MLFUtterance> m_utterances;
        std::vector<MLFFrame> m_frames;

        Timeline m_sequences;
        size_t m_elementSize;
        bool m_frameMode;
        std::wstring m_name;

    public:
        MLFDataDeserializer(const ConfigParameters& label, size_t elementSize, const HTKDataDeserializer* featureDeserializer, bool framMode, const std::wstring& featureName);

        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;

        virtual const Timeline& GetSequenceDescriptions() const override;

        virtual std::vector<InputDescriptionPtr> GetInputs() const override;

        virtual std::vector<std::vector<Sequence>> GetSequencesById(const std::vector<size_t> & ids) override;

        virtual bool RequireChunk(size_t chunkIndex) override;

        virtual void ReleaseChunk(size_t chunkIndex) override;

    public:
        const std::vector<MLFUtterance>& GetUtterances() const;
    };

    typedef std::shared_ptr<MLFDataDeserializer> MLFDataDeserializerPtr;
}}}