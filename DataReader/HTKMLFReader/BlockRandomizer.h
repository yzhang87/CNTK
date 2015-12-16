//
// <copyright file="BlockRandomizer.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include <vector>

#include "InnerInterfaces.h"
#include "minibatchsourcehelpers.h" // for msra::dbn::rand

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
        DataDeserializerPtr m_sequencer;
        size_t m_numSequences;
        size_t m_numChunks;
        size_t m_numSamples;
        bool m_frameMode; // true iff only single-sample sequences
        std::vector<ChunkInformation> m_chunkInformation;

        // Per-epoch configuration
        size_t m_workerRank;
        size_t m_numberOfWorkers;
        size_t m_epochSize;
        size_t m_samplePositionInEpoch;

        // Per-randomization-sweep information
        size_t m_sweep;
        size_t m_sweepStartInSamples; // TODO do we need it?
        size_t m_sequencePositionInSweep;
        std::vector<RandomizedChunk> m_randomizedChunks;
        std::vector<size_t> m_sequencePositionToChunkIndex; // TODO find on m_randomizedChunks instead?
        Timeline m_randomTimeline;

        bool IsValid(const TimelineP& timeline) const;

        template<typename VECTOR> static void randomShuffle(VECTOR & v, size_t randomseed);

        void RandomizeChunks();

        bool IsValidForPosition(size_t targetPosition, const SequenceDescription & seqDesc) const;

        void Randomize();

        void RandomizeForGlobalSamplePosition(const size_t samplePosition);

        void RandomizeIfNewSweepIsEntered();

        bool AdvanceToNextPositionForThisWorker();

    public:
        BlockRandomizer(int verbosity, size_t randomizationRangeInSamples, DataDeserializerPtr bundler);

        virtual ~BlockRandomizer() { }

        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;

        virtual std::vector<InputDescriptionPtr> GetInputs() const override
        {
            return m_sequencer->GetInputs();
        }

        virtual SequenceData GetNextSequence() override;
    };

} } }
