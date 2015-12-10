//
// <copyright file="BlockRandomizer.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "BlockRandomizer.h"
#include <algorithm>
#include <utility> // std::swap
#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    //
    // Private methods
    //

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
                    && 0 < current.numberOfSamples;
                previous = current;
                return result;
            });
        return it == timeline.end();
    }

    // Shuffle a vector into random order by randomly swapping elements
    template<typename VECTOR> static void BlockRandomizer::randomShuffle(VECTOR & v, size_t randomseed)
    {
        if (v.size() > RAND_MAX * static_cast<size_t>(RAND_MAX))
        {
            RuntimeError("randomShuffle: too large set: need to change to different random generator!");
        }
        srand(static_cast<unsigned int>(randomseed));
        foreach_index (i, v)
        {
            // pick a random location
            const size_t irand = msra::dbn::rand(0, v.size());

            // swap element i with it
            if (irand == static_cast<size_t>(i))
                continue;
            std::swap(v[i], v[irand]);
        }
    }

    void BlockRandomizer::RandomizeChunks(const size_t sweep, const size_t sweepts)
    {
        // Create vector of chunk indices and shuffle them using current sweep as seed
        std::vector<size_t> randomizedChunkIndices;
        randomizedChunkIndices.reserve(m_numChunks);
        for (size_t i = 0; i < m_numChunks; i++)
        {
            randomizedChunkIndices.push_back(i);
        }
        randomShuffle(randomizedChunkIndices, sweep);

        // Place randomized chunks on global time line
        m_randomizedChunks.clear();
        m_randomizedChunks.reserve(m_numChunks + 1);
        size_t chunkId, samplePosition, sequencePosition; // TODO xxx
        for (chunkId = 0, samplePosition = sweepts, sequencePosition = 0; chunkId < m_numChunks; chunkId++)
        {
            const size_t originalChunkIndex = randomizedChunkIndices[chunkId];
            const size_t numSequences =
                m_chunkInformation[originalChunkIndex + 1].sequencePositionStart -
                m_chunkInformation[originalChunkIndex].sequencePositionStart;
            const size_t numSamples =
                m_chunkInformation[originalChunkIndex + 1].samplePositionStart -
                m_chunkInformation[originalChunkIndex].samplePositionStart;
            m_randomizedChunks.push_back(RandomizedChunk {
                sequencePosition, samplePosition,
                originalChunkIndex
            });
            samplePosition += numSamples;
            sequencePosition += numSequences;
        }

        // Add sentinel
        m_randomizedChunks.push_back(RandomizedChunk {
            sequencePosition, samplePosition, SIZE_MAX
        });

        // For each chunk, compute the randomization range (w.r.t. the randomized chunk sequence)
        size_t halfWindowRange = m_randomizationRangeInSamples / 2;
        for (size_t chunkId = 0; chunkId < m_numChunks; chunkId++)
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
                chunk.windowbegin = m_randomizedChunks[chunkId - 1].windowbegin; // might be too early
                chunk.windowend = m_randomizedChunks[chunkId - 1].windowend; // might have more space
            }
            while (chunk.info.samplePositionStart - m_randomizedChunks[chunk.windowbegin].info.samplePositionStart > halfWindowRange)
                chunk.windowbegin++; // too early
            while (chunk.windowend < m_numChunks &&
                m_randomizedChunks[chunk.windowend + 1].info.samplePositionStart - chunk.info.samplePositionStart < halfWindowRange)
                chunk.windowend++; // got more space
        }

        // Compute the randomization range for sequence positions.
        m_sequencePositionToChunkIndex.clear();
        m_sequencePositionToChunkIndex.reserve(m_numSequences);
        for (size_t k = 0; k < m_numChunks; k++)
        {
            const size_t numSequences =
                m_randomizedChunks[k + 1].info.sequencePositionStart -
                m_randomizedChunks[k].info.sequencePositionStart;
            for (size_t i = 0; i < numSequences; i++)
            {
                m_sequencePositionToChunkIndex.push_back(k);
            }
        }
        assert(m_sequencePositionToChunkIndex.size() == m_numSequences);
    }

    bool BlockRandomizer::IsValidForPosition(size_t targetPosition, const SequenceDescription & seqDesc) const
    {
        const auto & chunk = m_randomizedChunks[m_sequencePositionToChunkIndex[targetPosition]];
        return chunk.windowbegin <= seqDesc.chunkId && seqDesc.chunkId < chunk.windowend;
    }

    void BlockRandomizer::Randomize(
        const size_t sweep,
        const size_t sweepts,
        const Timeline& timeline)
    {
        RandomizeChunks(sweep, sweepts);

        // Set up m_randomTimeline, shuffled by chunks.
        m_randomTimeline.clear();
        m_randomTimeline.reserve(m_numSequences);
        for (size_t chunkId = 0; chunkId < m_numChunks; chunkId++)
        {
            auto originalChunkIndex = m_randomizedChunks[chunkId].originalChunkIndex;

            for (size_t sequencePosition = m_chunkInformation[originalChunkIndex].sequencePositionStart;
                sequencePosition < m_chunkInformation[originalChunkIndex + 1].sequencePositionStart;
                sequencePosition++)
            {
                SequenceDescription randomizedSeqDesc = timeline[sequencePosition];
                randomizedSeqDesc.chunkId = chunkId;
                m_randomTimeline.push_back(randomizedSeqDesc);
            }
        }
        assert(m_randomTimeline.size() == m_numSequences);

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
            const size_t chunkId = m_sequencePositionToChunkIndex[i];
            const size_t windowbegin = m_randomizedChunks[chunkId].windowbegin;
            const size_t windowend = m_randomizedChunks[chunkId].windowend;

            // Get valid randomization range, expressed in sequence positions.
            size_t posbegin = m_randomizedChunks[windowbegin].info.sequencePositionStart;
            size_t posend = m_randomizedChunks[windowend].info.sequencePositionStart;

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
                std::swap(m_randomTimeline[i], m_randomTimeline[j]); // TODO old swap was perhaps more efficient
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

    void BlockRandomizer::LazyRandomize()
    {
        if (m_currentSequencePositionInSweep >= m_numSequences)
        {
            if (m_verbosity > 0)
                fprintf(stderr, "lazyrandomization: re-randomizing for sweep %llu in %s mode\n",
                    m_currentSweep, m_frameMode ? "frame" : "utterance");
            m_currentSweep++;
            Randomize(m_currentSweep, 0 /* TODO should not need it anymore? */, m_sequencer->GetTimeline());
            m_currentSequencePositionInSweep = 0;
        };
    }

    //
    // Public methods
    //

    BlockRandomizer::BlockRandomizer(int verbosity, size_t randomizationRangeInSamples, SequencerPtr sequencer)
        : m_verbosity(verbosity)
        , m_randomizationRangeInSamples(randomizationRangeInSamples)
        , m_sequencer(sequencer)
        , m_currentSweep(SIZE_MAX)
        , m_currentSequencePositionInSweep(SIZE_MAX)
        , m_currentSamplePositionInEpoch(SIZE_MAX)
        , m_epochSize(SIZE_MAX)
    {
        assert(sequencer != nullptr);
        const Timeline & timeline = m_sequencer->GetTimeline();
        assert(IsValid(timeline));

        m_numSequences = timeline.back().id + 1;
        m_numChunks = timeline.back().chunkId + 1;

        // Generate additional information about physical chunks
        assert(m_chunkInformation.size() == 0);
        m_chunkInformation.reserve(m_numChunks + 1);
        m_chunkInformation.insert(m_chunkInformation.begin(),
            m_numChunks + 1,
            ChunkInformation { SIZE_MAX, SIZE_MAX } );

        size_t maxNumberOfSamples = 0;

        m_numSamples = 0;
        for (const auto & seqDesc : timeline)
        {
            auto & chunkInformation = m_chunkInformation[seqDesc.chunkId];
            chunkInformation.sequencePositionStart =
                min(chunkInformation.sequencePositionStart, seqDesc.id);
            chunkInformation.samplePositionStart =
                min(chunkInformation.samplePositionStart, m_numSamples);
            maxNumberOfSamples = max(maxNumberOfSamples, seqDesc.numberOfSamples);
            m_numSamples += seqDesc.numberOfSamples;
        }

        // Add sentinel
        m_chunkInformation[m_numChunks] = { m_numSequences, m_numSamples };

        // Frame mode to the randomizer just means there are only single-sample sequences
        m_frameMode = (maxNumberOfSamples == 1);
    }

    void BlockRandomizer::SetEpochConfiguration(const EpochConfiguration& config)
    {
        m_sequencer->SetEpochConfiguration(config);

        size_t totalFrames = 0;
        for (const auto& t : m_sequencer->GetTimeline())
        {
            totalFrames += t.numberOfSamples;
        }

        // eldak: check partial minibatches.
        if (config.totalSize == requestDataSize)
        {
            m_epochSize = totalFrames;
        } 
        else
        {
            m_epochSize = config.totalSize;
        }

        // TODO add some asserts on EpochConfiguration
        m_config = config;
        m_currentSamplePositionInEpoch = 0;
        size_t timeframe = m_epochSize * config.index;

        assert(m_frameMode);

        // TODO make sure this will use the lazy path as well...
        m_currentSweep = timeframe / m_numSamples;
        Randomize(m_currentSweep, 0 /* TODO should not need it anymore? */, m_sequencer->GetTimeline());
        m_currentSequencePositionInSweep = timeframe % m_numSamples;
    };

    SequenceData BlockRandomizer::GetNextSequence()
    {
        assert(m_currentSamplePositionInEpoch != SIZE_MAX); // SetEpochConfiguration() must be called first
        if (m_currentSamplePositionInEpoch >= m_epochSize)
        {
            SequenceData result;
            result.m_endOfEpoch = true;
            return result;
        }

        LazyRandomize();
        assert(m_currentSequencePositionInSweep < m_numSequences);
        const auto & seqDesc = m_randomTimeline[m_currentSequencePositionInSweep];

        // Require and release chunks from the sequencer
        const auto & chunk = m_randomizedChunks[m_sequencePositionToChunkIndex[m_currentSequencePositionInSweep]];
        const size_t windowbegin = chunk.windowbegin;
        const size_t windowend = chunk.windowend;

        for (size_t chunkId = 0; chunkId < m_numChunks; chunkId++)
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

        m_currentSamplePositionInEpoch += seqDesc.numberOfSamples;
        m_currentSequencePositionInSweep++;

        return m_sequencer->GetSequenceById(seqDesc.id);
    };

} } }
