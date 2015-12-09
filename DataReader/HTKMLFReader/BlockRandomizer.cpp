//
// <copyright file="BlockRandomizer.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "BlockRandomizer.h"
#include <algorithm>
#include <DataReader.h>

namespace msra { namespace dbn {

    using namespace Microsoft::MSR::CNTK;

    // shuffle a vector into random order by randomly swapping elements
    template<typename VECTOR> static void BlockRandomizer::randomshuffle(VECTOR & v, size_t randomseed)
    {
        if (v.size() > RAND_MAX * (size_t) RAND_MAX)
        {
            RuntimeError("randomshuffle: too large set: need to change to different random generator!");
        }
        srand(static_cast<unsigned int>(randomseed));
        foreach_index (i, v)
        {
            // pick a random location
            const size_t irand = msra::dbn::rand (0, v.size());

            // swap element i with it
            if (irand == (size_t) i)
                continue;
            ::swap (v[i], v[irand]);
        }
    }

    void BlockRandomizer::InitializeChunkInformation()
    {
        const Timeline & timeline = m_sequencer->getTimeline();
        assert(IsValid(timeline));

        const size_t numChunks = timeline.back().chunkId + 1;
        assert(m_chunkInformation.size() == 0);

        m_chunkInformation.insert(m_chunkInformation.begin(),
            numChunks,
            ChunkInformation { 0, 0, SIZE_MAX } );

        size_t maxNumberOfSamples = 0;

        for (const auto & seqDesc : timeline)
        {
            auto & chunkInformation = m_chunkInformation[seqDesc.chunkId];
            chunkInformation.numSequences++;
            chunkInformation.numSamples += seqDesc.numberOfSamples;
            chunkInformation.sequencePositionStart =
                min(chunkInformation.sequencePositionStart, seqDesc.id);
            maxNumberOfSamples = max(maxNumberOfSamples, seqDesc.numberOfSamples);
        }

        assert(!m_framemode || maxNumberOfSamples == 1);
    }

    void BlockRandomizer::Randomize(
        const size_t sweep,
        const size_t sweepts, // TODO not needed anymore
        const Timeline& timeline)
    {
        // TODO make a members
        const size_t numChunks = timeline.back().chunkId + 1;
        const size_t numSequences = timeline.back().id + 1;

        // Create vector of chunk indices and shuffle them using current sweep as seed
        std::vector<size_t> randomizedChunkIndices;
        randomizedChunkIndices.reserve(numChunks);
        for (size_t i = 0; i < numChunks; i++)
        {
            randomizedChunkIndices.push_back(i);
        }
        randomshuffle(randomizedChunkIndices, sweep);

        // Place randomized chunks on global time line
        m_randomizedChunks.clear();
        m_randomizedChunks.reserve(numChunks);
        for (size_t chunkId = 0, t = sweepts /* TODO could drop */, pos = 0; chunkId < numChunks; chunkId++)
        {
            const size_t originalChunkIndex = randomizedChunkIndices[chunkId];
            const size_t numSequences = m_chunkInformation[originalChunkIndex].numSequences;
            const size_t numSamples = m_chunkInformation[originalChunkIndex].numSamples;
            m_randomizedChunks.push_back(RandomizedChunk(
                originalChunkIndex, // TODO this one still needed
                numSequences,
                numSamples,
                pos,
                t)); // TODO this one still needed
            t += numSamples;
            pos += numSequences;
        }

        // For each chunk, compute the randomization range (w.r.t. the randomized chunk sequence)
        foreach_index(chunkId, m_randomizedChunks)
        {
            auto & chunk = m_randomizedChunks[chunkId];
            // start with the range of left neighbor
            if (chunkId == 0)
            {
                chunk.windowbegin = 0;
                chunk.windowend = 1;
            }
            else
            {
                chunk.windowbegin = m_randomizedChunks[chunkId - 1].windowbegin;  // might be too early
                chunk.windowend = m_randomizedChunks[chunkId - 1].windowend;      // might have more space
            }
            while (chunk.globalSamplePositionStart - m_randomizedChunks[chunk.windowbegin].globalSamplePositionStart > m_randomizationrange / 2)
                chunk.windowbegin++;            // too early
            while (chunk.windowend < numChunks &&
                m_randomizedChunks[chunk.windowend].globalSamplePositionEnd() - chunk.globalSamplePositionStart < m_randomizationrange / 2)
                chunk.windowend++;              // got more space
        }

        // Compute the randomization range for sequence positions.
        // TODO just map position to randomized chunk index
        positionchunkwindows.clear();
        positionchunkwindows.reserve(numSequences);
        foreach_index (k, m_randomizedChunks)
        {
            const auto & chunk = m_randomizedChunks[k];
            for (size_t i = 0; i < chunk.numSequences; i++)
            {
                positionchunkwindows.push_back(m_randomizedChunks.begin() + k);
            }
        }
        assert(positionchunkwindows.size() == numSequences);

        // Set up m_randomTimeline, shuffled by chunks.
        m_randomTimeline.clear();
        m_randomTimeline.reserve(numSequences);
        foreach_index (chunkId, m_randomizedChunks)
        {
            const auto & chunk = m_randomizedChunks[chunkId];

            // TODO pos -> iterator
            for (size_t i = 0, pos = m_chunkInformation[chunk.originalChunkIndex].sequencePositionStart; i < chunk.numSequences; i++, pos++)
            {
                SequenceDescription randomizedSeqDesc = timeline[pos];
                randomizedSeqDesc.chunkId = chunkId;
                m_randomTimeline.push_back(randomizedSeqDesc);
            }
        }
        assert(m_randomTimeline.size() == numSequences);

        // Check we got those setup right
        foreach_index (i, m_randomTimeline)
        {
            assert(IsValidForPosition(i, m_randomTimeline[i]));
        }

        // Now randomly shuffle m_randomTimeline, while considering the
        // constraints of what chunk range needs to be in memory.
        srand(static_cast<unsigned int>(sweep + 1));
        foreach_index (i, m_randomTimeline)
        {
            // Get valid randomization range, expressed in chunks
            const size_t windowbegin = positionchunkwindows[i].windowbegin();
            const size_t windowend = positionchunkwindows[i].windowend();

            // Get valid randomization range, expressed in sequence positions.
            size_t posbegin = m_randomizedChunks[windowbegin].sequencePositionStart;
            size_t posend = m_randomizedChunks[windowend - 1].sequencePositionEnd();

            for (;;)
            {
                // Pick a sequence position from [posbegin, posend)
                const size_t j = msra::dbn::rand(posbegin, posend);

                // Try again if the sequence currently at j cannot be placed at position i.
                if (!IsValidForPosition(i, m_randomTimeline[j]))
                    continue;

                // Try again if the sequence currently at i cannot be placed at position j.
                if (!IsValidForPosition(j, m_randomTimeline[i]))
                    continue;

                // Swap and break out.
                ::swap (m_randomTimeline[i], m_randomTimeline[j]); // TODO old swap was perhaps more efficient
                break;
            }
        }

        // Verify that we got it right
        foreach_index (i, m_randomTimeline)
        {
            // TODO assert only
            if (!IsValidForPosition(i, m_randomTimeline[i]))
                LogicError("lazyrandomization: randomization logic mangled!");
        }
    }

    bool BlockRandomizer::IsValidForPosition(size_t targetPosition, const SequenceDescription & seqDesc) const
    {
        return positionchunkwindows[targetPosition].isvalidforthisposition(seqDesc);
    }

    bool BlockRandomizer::IsValid(const Timeline& timeline) const
    {
        SequenceDescription previous = {
            static_cast<size_t>(-1),
            0,
            0
        };
        auto it = std::find_if_not(timeline.begin(), timeline.end(),
            [&](const SequenceDescription& current)
            {
                bool result = previous.id + 1 == current.id
                    && previous.chunkId <= current.chunkId
                    && current.chunkId <= previous.chunkId + 1
                    && 0 < current.numberOfSamples
                    && (!m_framemode || current.numberOfSamples == 1);
                previous = current;
                return result;
            });
        return it == timeline.end();
    }

    void BlockRandomizer::LazyRandomize()
    {
        if (m_currentSequenceId >= m_randomTimeline.size())
        {
            if (m_verbosity > 0)
                fprintf(stderr, "lazyrandomization: re-randomizing for sweep %llu in %s mode\n",
                    m_currentSweep, m_framemode ? "frame" : "utterance");
            m_currentSweep++;
            Randomize(m_currentSweep, 0 /* TODO should not need it anymore? */, m_sequencer->getTimeline());
            m_currentSequenceId = 0;
        };
    }

    SequenceData BlockRandomizer::getNextSequence()
    {
        assert(m_currentFrame != SIZE_MAX); // SetEpochConfiguration() must be called first
        if (m_currentFrame >= m_epochSize)
        {
            SequenceData result;
            result.m_endOfEpoch = true;
            return result;
        }

        LazyRandomize();
        assert(m_currentSequenceId < m_randomTimeline.size());
        const auto & seqDesc = m_randomTimeline[m_currentSequenceId];

        // Require and release chunks from the sequencer
        const size_t windowbegin = getSequenceWindowBegin(m_currentSequenceId);
        const size_t windowend = getSequenceWindowEnd(m_currentSequenceId);
        const size_t numChunks = m_sequencer->getTimeline().back().chunkId + 1;

        for (size_t chunkId = 0; chunkId < numChunks; chunkId++)
        {
            auto originalChunkIndex = m_randomizedChunks[chunkId].originalChunkIndex;

            if (windowbegin <= chunkId && chunkId < windowend)
            {
                // TODO missing: for distributed case only need some of the chunks
                m_sequencer->RequireChunk(originalChunkIndex);
            }
            else
            {
                m_sequencer->ReleaseChunk(originalChunkIndex);
            }
        }

        m_currentFrame += seqDesc.numberOfSamples;
        m_currentSequenceId++;

        return m_sequencer->getSequenceById(seqDesc.id);
    };

    void BlockRandomizer::SetEpochConfiguration(const EpochConfiguration& config)
    {
        m_config = config;
        m_currentFrame = 0;
        m_epochSize = config.totalSize;
        size_t timeframe = m_epochSize * config.index;

        size_t totalSize = 0;
        for (const auto& t : m_sequencer->getTimeline())
        {
            totalSize += t.numberOfSamples;
        }

        assert(m_framemode);

        // TODO make sure this will use the lazy path as well...
        m_currentSweep = timeframe / totalSize;
        Randomize(m_currentSweep, 0 /* TODO should not need it anymore? */, m_sequencer->getTimeline());
        m_currentSequenceId = timeframe % totalSize;
    };
} }
