//
// <copyright file="BlockRandomizer.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "minibatchsourcehelpers.h" // for msra::dbn::rand
#include "unordered_set"
#include "InnerInterfaces.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class BlockRandomizer : public Transformer
    {
        // Structure for per-chunk information
        struct ChunkInformation
        {
            size_t sequencePositionStart;
            size_t samplePositionStart;
        };

        // Structure that will be maintained for each randomized chunk
        struct RandomizedChunk
        {
            struct ChunkInformation info; // sample positions are global // TODO could drop 'global' requirement?

            size_t originalChunkIndex;

            // Randomization range (in randomized chunk positions)
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
        bool m_frameMode; // true iff only single-sample sequences
        std::vector<ChunkInformation> m_chunkInformation;

        // Per-epoch configuration
        EpochConfiguration m_config;
        size_t m_epochSize;
        size_t m_currentSamplePositionInEpoch;

        // Per-randomization-sweep information
        size_t m_currentSweep;
        size_t m_currentSequencePositionInSweep; // position within the current sweep
        std::vector<RandomizedChunk> m_randomizedChunks;
        std::vector<size_t> m_sequencePositionToChunkIndex; // TODO find on m_randomizedChunks instead?
        Timeline m_randomTimeline;

        bool IsValid(const Timeline& timeline) const;

        template<typename VECTOR> static void randomShuffle(VECTOR & v, size_t randomseed);

        void RandomizeChunks(const size_t sweep, const size_t sweepts); // TODO drop sweepts? // TODO rename?

        bool IsValidForPosition(size_t targetPosition, const SequenceDescription & seqDesc) const;

        void Randomize(const size_t sweep, const size_t sweepts, const Timeline& timeline); // TODO drop sweepts? 

        void LazyRandomize();

    public:
        BlockRandomizer(int verbosity, size_t randomizationRangeInSamples, SequencerPtr sequencer);

        virtual ~BlockRandomizer() { }

        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;

        virtual std::vector<InputDescriptionPtr> GetInputs() const override
        {
            return m_sequencer->GetInputs();
        }

        virtual SequenceData GetNextSequence() override;
    };

} } }
