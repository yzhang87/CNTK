//
// <copyright file="BlockRandomizer.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// BlockRandomizer.h -- interface of the block randomizer
//

#pragma once

#include "Basics.h"                  // for attempt()
#include "htkfeatio.h"                  // for htkmlfreader
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "minibatchsourcehelpers.h"
#include "minibatchiterator.h"
#include "biggrowablevectors.h"
#include "ssematrix.h"
#include "unordered_set"
#include "inner_interfaces.h"

namespace msra { namespace dbn {

    using namespace Microsoft::MSR::CNTK;

    class BlockRandomizer : public Transformer
    {
        // Structure for per-chunk information
        // TODO note: numSequences / numSamples could also be computed through neighbors
        struct ChunkInformation
        {
            size_t numSequences;
            size_t numSamples;
            size_t sequencePositionStart;
            size_t samplePositionStart;

            size_t sequencePositionEnd() const { return sequencePositionStart + numSequences; }
            size_t samplePositionEnd() const { return samplePositionStart + numSamples; }
        };

        // Structure that will be maintained for each randomized chunk
        struct RandomizedChunk
        {
            struct ChunkInformation info; // sample positions are global // TODO could drop?

            size_t originalChunkIndex;

            // Randomization range limits (randomized chunk positions)
            size_t windowbegin;
            size_t windowend;
        };

        // General configuration
        int m_verbosity;
        size_t m_randomizationRangeInSamples; // full window

        // Sequencer and information on the original timeline
        SequencerPtr m_sequencer;
        size_t m_numSequences;
        size_t m_numChunks;
        size_t m_numSamples;
        bool m_framemode; // true iff only single-sample sequences
        std::vector<ChunkInformation> m_chunkInformation;

        // Per-epoch configuration
        EpochConfiguration m_config;
        size_t m_epochSize;
        size_t m_currentSamplePositionInEpoch;

        // Per-randomization-sweep information
        size_t m_currentSweep;
        size_t m_currentSequencePositionInSweep; // position within the current sweep
        std::vector<RandomizedChunk> m_randomizedChunks;
        std::vector<size_t> m_sequencePositionToChunkIndex;
        Timeline m_randomTimeline;

        template<typename VECTOR> static void randomshuffle(VECTOR & v, size_t randomseed);

        bool IsValid(const Timeline& timeline) const;

        void RandomizeChunks(const size_t sweep, const size_t sweepts); // TODO drop sweepts?

        bool IsValidForPosition(size_t targetPosition, const SequenceDescription & seqDesc) const;

        void Randomize(
            const size_t sweep,
            const size_t sweepts, // TODO drop sweepts?
            const Timeline& timeline);

        void LazyRandomize();

        size_t getSequenceWindowBegin(size_t sequenceIndex) const
        {
            assert(sequenceIndex < m_numSequences);
            const auto & chunk = m_randomizedChunks[m_sequencePositionToChunkIndex[sequenceIndex]];
            return chunk.windowbegin;
        }

        size_t getSequenceWindowEnd(size_t sequenceIndex) const
        {
            assert(sequenceIndex < m_numSequences);
            const auto & chunk = m_randomizedChunks[m_sequencePositionToChunkIndex[sequenceIndex]];
            return chunk.windowend;
        }

    public:
        BlockRandomizer(int verbosity, size_t randomizationRangeInSamples, SequencerPtr sequencer);

        virtual ~BlockRandomizer()
        {
        }

        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;

        virtual std::vector<InputDescriptionPtr> getInputs() const override
        {
            std::vector<InputDescriptionPtr> dummy;
            return dummy;
        }

        virtual SequenceData getNextSequence() override;
    };
} }
