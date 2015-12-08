//
// <copyright file="BlockRandomizer.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// BlockRandomizer.cpp -- implementation of the block randomizer
//

#include "stdafx.h"
#include "BlockRandomizer.h"
#include <algorithm>
#include <DataReader.h>

namespace msra { namespace dbn {

    using namespace Microsoft::MSR::CNTK;

    // shuffle a vector into random order by randomly swapping elements
    template<typename VECTOR> static void BlockRandomizer::randomshuffle (VECTOR & v, size_t randomseed)
    {
        if (v.size() > RAND_MAX * (size_t) RAND_MAX)
            RuntimeError("randomshuffle: too large set: need to change to different random generator!");
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
            chunkInformation.startSequencePosition =
                min(chunkInformation.startSequencePosition, seqDesc.id);
            maxNumberOfSamples = max(maxNumberOfSamples, seqDesc.numberOfSamples);
        }

        assert(!m_framemode || maxNumberOfSamples == 1);
    }

    void BlockRandomizer::Randomize(
        const size_t sweep,
        const size_t sweepts, // TODO not needed anymore
        const Timeline& timeline)
    {
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
        randomizedchunks.clear();
        randomizedchunks.reserve(numChunks);
        for (size_t chunkId = 0, t = sweepts /* TODO could drop */, pos = 0; chunkId < numChunks; chunkId++)
        {
            const size_t originalChunkIndex = randomizedChunkIndices[chunkId];
            const size_t numutterances = m_chunkInformation[originalChunkIndex].numSequences;
            const size_t numframes = m_chunkInformation[originalChunkIndex].numSamples;
            randomizedchunks.push_back(chunk(
                originalChunkIndex, // TODO this one still needed
                numutterances,
                numframes,
                pos,
                t)); // TODO this one still needed
            t += numframes;
            pos += numutterances;
        }

        // For each chunk, compute the randomization range (w.r.t. the randomized chunk sequence)
        foreach_index(chunkId, randomizedchunks)
        {
            chunk & chunk = randomizedchunks[chunkId];
            // start with the range of left neighbor
            if (chunkId == 0)
            {
                chunk.windowbegin = 0;
                chunk.windowend = 1;
            }
            else
            {
                chunk.windowbegin = randomizedchunks[chunkId - 1].windowbegin;  // might be too early
                chunk.windowend = randomizedchunks[chunkId - 1].windowend;      // might have more space
            }
            while (chunk.globalts - randomizedchunks[chunk.windowbegin].globalts > m_randomizationrange / 2)
                chunk.windowbegin++;            // too early
            while (chunk.windowend < numChunks &&
                randomizedchunks[chunk.windowend].globalte() - chunk.globalts < m_randomizationrange / 2)
                chunk.windowend++;              // got more space
        }

        // Compute the randomization range for sequence positions.
        // TODO just map position to randomized chunk index
        positionchunkwindows.clear();
        positionchunkwindows.reserve(numSequences);
        foreach_index (k, randomizedchunks)
        {
            chunk & chunk = randomizedchunks[k];
            for (size_t i = 0; i < chunk.numutterances; i++)
            {
                positionchunkwindows.push_back(randomizedchunks.begin() + k);
            }
        }
        assert(positionchunkwindows.size() == numSequences);

        // Set up m_randomTimeline, shuffled by chunks.
        m_randomTimeline.clear();
        m_randomTimeline.reserve(numSequences);
        for (const auto & chunk : randomizedchunks)
        {
            // TODO pos -> iterator
            for (size_t i = 0, pos = m_chunkInformation[chunk.originalChunkIndex].startSequencePosition; i < chunk.numutterances; i++, pos++)
            {
                m_randomTimeline.push_back(timeline[pos]);
            }
        }
        assert(m_randomTimeline.size() == numSequences);

        // Check we got those setup right
        foreach_index (i, m_randomTimeline)
        {
            assert(positionchunkwindows[i].isvalidforthisposition(m_randomTimeline[i]));
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
            size_t posbegin = randomizedchunks[windowbegin].utteranceposbegin;
            size_t posend = randomizedchunks[windowend - 1].utteranceposend();

            for(;;)
            {
                // Pick a sequence position from [posbegin, posend)
                const size_t j = msra::dbn::rand(posbegin, posend);

                // Try again if the sequence currently at j cannot be placed at position i.
                if (!positionchunkwindows[i].isvalidforthisposition(m_randomTimeline[j]))
                    continue;

                // Try again if the sequence currently at i cannot be placed at position j.
                if (!positionchunkwindows[j].isvalidforthisposition(m_randomTimeline[i]))
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
            if (!positionchunkwindows[i].isvalidforthisposition(m_randomTimeline[i]))
                LogicError("lazyrandomization: randomization logic mangled!");
        }
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
        if (m_currentFrame >= m_epochSize)
        {
            SequenceData result;
            result.m_endOfEpoch = true;
            return result;
        }

        LazyRandomize();
        assert(m_currentSequenceId < m_randomTimeline.size());
        const auto & seqDesc = m_randomTimeline[m_currentSequenceId++];
        m_currentFrame += seqDesc.numberOfSamples;

        // TODO purge maybe
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
