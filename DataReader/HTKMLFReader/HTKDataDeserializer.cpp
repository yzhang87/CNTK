#include "stdafx.h"
#include "HTKDataDeserializer.h"
#include "ConfigHelper.h"
#include "BundlerSplitted.h"
#include <numeric>

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

        size_t totalSize = std::accumulate(
            m_sequences.begin(),
            m_sequences.end(),
            static_cast<size_t>(0),
            [](size_t sum, const HTKSequenceDescription& s) {
                return s.numberOfSamples + sum;
        });

        // distribute them over chunks
        // We simply count off frames until we reach the chunk size.
        // Note that we first randomize the chunks, i.e. when used, chunks are non-consecutive and thus cause the disk head to seek for each chunk.
        const size_t framespersec = 100;                    // we just assume this; our efficiency calculation is based on this
        const size_t chunkframes = 15 * 60 * framespersec;  // number of frames to target for each chunk

        // Loading an initial 24-hour range will involve 96 disk seeks, acceptable.
        // When paging chunk by chunk, chunk size ~14 MB.

        m_chunks.resize(0);
        m_chunks.reserve(totalSize / chunkframes);

        foreach_index(i, m_sequences)
        {
            // if exceeding current entry--create a new one
            // I.e. our chunks are a little larger than wanted (on av. half the av. utterance length).
            if (m_chunks.empty() || m_chunks.back().totalframes > chunkframes || m_chunks.back().numutterances() >= 65535)
            {
                // TODO > instead of >= ? if (thisallchunks.empty() || thisallchunks.back().totalframes > chunkframes || thisallchunks.back().numutterances() >= frameref::maxutterancesperchunk)
                m_chunks.push_back(chunkdata());
            }

            // append utterance to last chunk
            chunkdata & currentchunk = m_chunks.back();
            currentchunk.push_back(&m_sequences[i].utterance);    // move it out from our temp array into the chunk
            // TODO: above push_back does not actually 'move' because the internal push_back does not accept that
        }

        fprintf(stderr, "minibatchutterancesource: %llu utterances grouped into %llu chunks, av. chunk size: %.1f utterances, %.1f frames\n",
            m_sequences.size(), m_chunks.size(), m_sequences.size() / (double)m_chunks.size(), totalSize / (double)m_chunks.size());
        // Now utterances are stored exclusively in allchunks[]. They are never referred to by a sequential utterance id at this point, only by chunk/within-chunk index.
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

    bool HTKDataDeserializer::RequireChunk(size_t chunkIndex)
    {
        auto & chunkdata = m_chunks[chunkIndex];
        if (chunkdata.isinram())
        {
            return false;
        }

        msra::util::attempt(5, [&]()   // (reading from network)
        {
            std::unordered_map<std::string, size_t> empty;
            msra::dbn::latticesource lattices(
                std::pair<std::vector<std::wstring>, std::vector<std::wstring>>(),
                empty);
            chunkdata.requiredata(m_featKind, m_featdim, m_sampperiod, lattices, m_verbosity);
        });

        m_chunksinram++;
        return true;
    }

    void HTKDataDeserializer::ReleaseChunk(size_t chunkIndex)
    {
        auto & chunkdata = m_chunks[chunkIndex];
        if (chunkdata.isinram())
        {
            chunkdata.releasedata();
            m_chunksinram--;
        }
    }
}}}
