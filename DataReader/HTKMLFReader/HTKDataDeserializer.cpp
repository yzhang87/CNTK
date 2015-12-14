#include "stdafx.h"
#include "HTKDataDeserializer.h"
#include "ConfigHelper.h"
#include "BundlerSplitted.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    HTKDataDeserializer::HTKDataDeserializer(const ConfigParameters& feature)
        : m_featureFiles(std::move(ConfigHelper::GetFeaturePaths(feature)))
    {
        ConfigHelper::CheckFeatureType(feature);

        auto context = ConfigHelper::GetContextWindow(feature);

        m_dimension = feature(L"dim");
        m_dimension = m_dimension * (1 + context.first + context.second);

        m_layout = std::make_shared<ImageLayout>(std::move(std::vector<size_t>{ m_dimension }));

        size_t numSequences = m_featureFiles.size();

        m_sequences.reserve(numSequences);
        m_sequencesP.reserve(numSequences);

        foreach_index(i, m_featureFiles)
        {
            utterancedesc utterance(msra::asr::htkfeatreader::parsedpath(m_featureFiles[i]), 0);
            const size_t uttframes = utterance.numframes(); // will throw if frame bounds not given --required to be given in this mode

            HTKSequenceDescription description(std::move(utterance));
            description.id = i;
            // description.chunkId, description.key // TODO

            // we need at least 2 frames for boundary markers to work
            if (uttframes < 2)
            {
                fprintf(stderr, "minibatchutterancesource: skipping %d-th file (%llu frames) because it has less than 2 frames: %ls\n",
                    i, uttframes, utterance.key().c_str());
                description.numberOfSamples = 0;
                description.isValid = false;
            }
            else
            {
                description.numberOfSamples = uttframes;
                description.isValid = true;
            }

            m_sequences.push_back(description);
            m_sequencesP.push_back(&m_sequences[i]);
        }
    }

    void Microsoft::MSR::CNTK::HTKDataDeserializer::SetEpochConfiguration(const EpochConfiguration& /*config*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    TimelineP Microsoft::MSR::CNTK::HTKDataDeserializer::GetSequenceDescriptions() const
    {
        return m_sequencesP;
    }

    Microsoft::MSR::CNTK::InputDescriptionPtr Microsoft::MSR::CNTK::HTKDataDeserializer::GetInput() const
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    Microsoft::MSR::CNTK::Sequence Microsoft::MSR::CNTK::HTKDataDeserializer::GetSequenceById(size_t /*id*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    Microsoft::MSR::CNTK::Sequence Microsoft::MSR::CNTK::HTKDataDeserializer::GetSampleById(size_t /*sequenceId*/, size_t /*sampleId*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    bool Microsoft::MSR::CNTK::HTKDataDeserializer::RequireChunk(size_t /*chunkIndex*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    void Microsoft::MSR::CNTK::HTKDataDeserializer::ReleaseChunk(size_t /*chunkIndex*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }
}}}
