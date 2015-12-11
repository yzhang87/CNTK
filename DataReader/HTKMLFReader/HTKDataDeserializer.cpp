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

        std::vector<std::wstring> featurePaths(ConfigHelper::GetFeaturePaths(feature));

        std::vector<bool> isValid(featurePaths.size(), true);
        std::vector<size_t> duration(featurePaths.size(), 0);

        foreach_index(i, featurePaths)
        {
            utterancedesc utterance(msra::asr::htkfeatreader::parsedpath(featurePaths[i]), 0);
            const size_t uttframes = utterance.numframes(); // will throw if frame bounds not given --required to be given in this mode

            // we need at least 2 frames for boundary markers to work
            if (uttframes < 2 || uttframes > 65535 /* TODO frameref::maxframesperutterance */)
            {
                fprintf(stderr, "minibatchutterancesource: skipping %llu-th file (%llu frames) because it exceeds max. frames (%llu) for frameref bit field: %ls\n", i, uttframes, 65535 /* frameref::maxframesperutterance */, utterance.key().c_str());
                duration[i] = 0;
                isValid[i] = false;
            }
            else
            {
                duration[i] = uttframes;
            }

            HTKSequenceDescription description(std::move(utterance));
            description.id = i;
            description.numberOfSamples = uttframes;

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