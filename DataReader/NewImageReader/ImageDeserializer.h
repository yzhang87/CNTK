#pragma once

#include "InnerInterfaces.h"
#include "commandArgUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageDataDeserializer : public DataDeserializer
    {
        struct ImageSequenceDescription : public SequenceDescription
        {
            std::string path;
            size_t classId;
        };
        std::vector<ImageSequenceDescription> m_imageSequences;

        TimelineP m_sequences;

        std::wstring m_featName;
        std::wstring m_labName;

        size_t m_featDim;
        size_t m_labDim;

        //using StrIntPairT = std::pair<std::string, int>;
        //std::vector<StrIntPairT> files;

        size_t m_epochSize;
        size_t m_mbSize;
        size_t m_epoch;

        size_t m_epochStart;
        size_t m_mbStart;
        //std::vector<ElemType> m_featBuf;
        //std::vector<ElemType> m_labBuf;

        bool m_imgListRand;
        size_t m_elementSize;
        int m_imgChannels;
        //MBLayoutPtr m_pMBLayout;

    public:
        ImageDataDeserializer(const ConfigParameters& config, size_t elementSize);
        std::vector<InputDescriptionPtr> GetInputs() const override;
        void SetEpochConfiguration(const EpochConfiguration& config) override;
        const TimelineP& GetSequenceDescriptions() const override;
        std::vector<Sequence> GetSequenceById(size_t id) override;
        bool RequireChunk(size_t chunkIndex) override;
        void ReleaseChunk(size_t chunkIndex) override;
    };
}}}