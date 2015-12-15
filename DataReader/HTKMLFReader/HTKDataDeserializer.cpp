#include "stdafx.h"
#include "HTKDataDeserializer.h"
#include "ConfigHelper.h"
#include "BundlerSplitted.h"
#include <numeric>

namespace Microsoft { namespace MSR { namespace CNTK {

    HTKDataDeserializer::HTKDataDeserializer(const ConfigParameters& feature, size_t elementSize)
        : m_featureFiles(std::move(ConfigHelper::GetFeaturePaths(feature)))
        , m_elementSize(elementSize)
    {
        ConfigHelper::CheckFeatureType(feature);

        auto context = ConfigHelper::GetContextWindow(feature);

        m_dimension = feature(L"dim");
        m_dimension = m_dimension * (1 + context.first + context.second);

        m_layout = std::make_shared<ImageLayout>(std::move(std::vector<size_t>{ m_dimension }));

        size_t numSequences = m_featureFiles.size();

        m_sequences.reserve(numSequences);
        m_sequencesP.reserve(numSequences);

        m_context = ConfigHelper::GetContextWindow(feature);

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
            m_sequences[i].indexInsideChunk = currentchunk.numutterances();
            currentchunk.push_back(&m_sequences[i].utterance);    // move it out from our temp array into the chunk
            
            // TODO: above push_back does not actually 'move' because the internal push_back does not accept that
        }

        fprintf(stderr, "minibatchutterancesource: %llu utterances grouped into %llu chunks, av. chunk size: %.1f utterances, %.1f frames\n",
            m_sequences.size(), m_chunks.size(), m_sequences.size() / (double)m_chunks.size(), totalSize / (double)m_chunks.size());
        // Now utterances are stored exclusively in allchunks[]. They are never referred to by a sequential utterance id at this point, only by chunk/within-chunk index.
    }

    void HTKDataDeserializer::SetEpochConfiguration(const EpochConfiguration& /*config*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    TimelineP HTKDataDeserializer::GetSequenceDescriptions() const
    {
        return m_sequencesP;
    }

    InputDescriptionPtr HTKDataDeserializer::GetInput() const
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    Sequence HTKDataDeserializer::GetSequenceById(size_t /*id*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    class matrixasvectorofvectors  // wrapper around a matrix that views it as a vector of column vectors
    {
        void operator= (const matrixasvectorofvectors &);  // non-assignable
        msra::dbn::matrixbase & m;
    public:
        matrixasvectorofvectors(msra::dbn::matrixbase & m) : m(m) {}
        size_t size() const { return m.cols(); }
        const_array_ref<float> operator[] (size_t j) const { return array_ref<float>(&m(0, j), m.rows()); }
    };

    Sequence HTKDataDeserializer::GetSampleById(size_t sequenceId, size_t sampleId)
    {
        msra::dbn::matrix feat;

        const std::vector<char> noboundaryflags;    // dummy

        const size_t spos = sampleId; // positer->second;
        const size_t epos = spos + 1;

        // Note that the above loop loops over all chunks incl. those that we already should have.
        // This has an effect, e.g., if 'numsubsets' has changed (we will fill gaps).

        // determine the true #frames we return, for allocation--it is less than mbframes in the case of MPI/data-parallel sub-set mode
        size_t tspos = 1; // eldak: what about parallel mode?


        feat.resize(m_dimension, tspos);

        //// return these utterances
        //if (verbosity > 0)
        //    fprintf(stderr, "getbatch: getting utterances %d..%d (%d subset of %d frames out of %d requested) in sweep %d\n", (int)spos, (int)(epos - 1), (int)tspos, (int)mbframes, (int)framesrequested, (int)sweep);
        tspos = 0;   // relative start of utterance 'pos' within the returned minibatch
        size_t numberOfFrames = 0;
        for (size_t pos = spos; pos < epos; pos++)
        {
            const auto& sequence = m_sequences[sampleId];

            size_t n = 0;
            const auto & chunkdata = m_chunks[sequence.chunkId];
            size_t dimension = m_dimension;

            auto uttframes = chunkdata.getutteranceframes(sequence.indexInsideChunk);
            matrixasvectorofvectors uttframevectors(uttframes);    // (wrapper that allows m[j].size() and m[j][i] as required by augmentneighbors())
            n = uttframevectors.size();


            size_t leftextent, rightextent;
            // page in the needed range of frames
            if (m_context.first == 0 && m_context.second == 0)
            {
                leftextent = rightextent = msra::dbn::augmentationextent(uttframevectors[0].size(), dimension);
            }
            else
            {
                leftextent = m_context.first;
                rightextent = m_context.second;
            }

            msra::dbn::augmentneighbors(uttframevectors, noboundaryflags, sampleId, leftextent, rightextent, feat, tspos);


            // copy the frames and class labels
            tspos += sequence.numberOfSamples;
            numberOfFrames++;
        }

        Sequence r;
        r.description = &m_sequences[sequenceId];

        const msra::dbn::matrixstripe featOri = msra::dbn::matrixstripe(feat, 0, feat.cols());
        const size_t dimensions = featOri.rows();
        const void* tmp = &featOri(0, 0);

        r.numberOfSamples = 1;

        // eldak: this should not be allocated each time.
        void* buffer = nullptr;
        if (m_elementSize == sizeof(float))
        {
            buffer = new float[featOri.rows()];
        }
        else
        {
            buffer = new double[featOri.rows()];
        }

        memset(buffer, 0, m_elementSize * dimensions);
        memcpy_s(buffer, m_elementSize * dimensions, tmp, m_elementSize * dimensions);
        r.data = buffer;

        return r;
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
