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

    class BlockRandomizer : public Microsoft::MSR::CNTK::Transformer
    {
        int m_verbosity;
        bool m_framemode; // TODO drop?
        Microsoft::MSR::CNTK::EpochConfiguration m_config;
        size_t m_currentFrame;
        size_t m_epochSize;
        Microsoft::MSR::CNTK::SequencerPtr m_sequencer;
        // Information maintained for original (non-randomized) chunks
        struct ChunkInformation
        {
            size_t numSequences;
            size_t numSamples;
            size_t sequencePositionStart;
        };
        std::vector<ChunkInformation> m_chunkInformation;
        size_t m_randomizationrange; // full window measured in samples, one half to the left, the other to the right
        size_t m_currentSweep; // randomization is currently cached for this sweep; if it changes, rebuild all below
        size_t m_currentSequenceId; // position within the current sweep

        // TODO note: numSequences / numSamples could also be computed through neighbors
        struct RandomizedChunk          // chunk as used in actual processing order (randomized sequence)
        {
            size_t originalChunkIndex;
            size_t numSequences;
            size_t numSamples;

            // Sequence positions (on randomized timeline) that this chunk covers
            size_t sequencePositionStart;
            size_t sequencePositionEnd() const { return sequencePositionStart + numSequences; }

            // (Global) Sample position (on randomized timeline)
            size_t globalSamplePositionStart;
            size_t globalSamplePositionEnd() const { return globalSamplePositionStart + numSamples; }
            // TODO only needed for window computation, change and drop that

            // Randomization range limits (randomized chunk positions)
            size_t windowbegin;
            size_t windowend;

            RandomizedChunk(size_t originalChunkIndex,
                size_t numSequences,
                size_t numSamples,
                size_t sequencePositionStart,
                size_t globalSamplePositionStart)
                : originalChunkIndex(originalChunkIndex)
                , numSequences(numSequences)
                , numSamples(numSamples)
                , sequencePositionStart(sequencePositionStart)
                , globalSamplePositionStart(globalSamplePositionStart) {}
        };
        std::vector<RandomizedChunk> m_randomizedChunks; // chunks after being brought into random order (we randomize within a rolling window over them)
        Microsoft::MSR::CNTK::Timeline m_randomTimeline;

        struct positionchunkwindow       // chunk window required in memory when at a certain position, for controlling paging
        {
            std::vector<RandomizedChunk>::iterator definingchunk;       // the chunk in m_randomizedChunks[] that defined the utterance position of this utterance
            size_t windowbegin() const { return definingchunk->windowbegin; }
            size_t windowend() const { return definingchunk->windowend; }

            bool isvalidforthisposition (const Microsoft::MSR::CNTK::SequenceDescription & sequence) const
            {
                return sequence.chunkId >= windowbegin() && sequence.chunkId < windowend(); // check if 'sequence' lives in is in allowed range for this position
                // TODO by construction sequences cannot span chunks (check again)
            }

            positionchunkwindow(std::vector<RandomizedChunk>::iterator definingchunk) : definingchunk(definingchunk) {}
        };
        std::vector<positionchunkwindow> positionchunkwindows;      // [utterance position] -> [windowbegin, windowend) for controlling paging
        // TODO improve, use randomized timeline?

        template<typename VECTOR> static void randomshuffle(VECTOR & v, size_t randomseed);

        void InitializeChunkInformation();

        bool IsValidForPosition(size_t targetPosition, const Microsoft::MSR::CNTK::SequenceDescription & seqDesc) const;

        bool IsValid(const Microsoft::MSR::CNTK::Timeline& timeline) const;

        void LazyRandomize();

        void Randomize(
            const size_t sweep,
            const size_t sweepts,
            const Microsoft::MSR::CNTK::Timeline& timeline);

    public:
        BlockRandomizer(int verbosity, bool framemode /* TODO drop */, size_t randomizationrange, Microsoft::MSR::CNTK::SequencerPtr sequencer)
            : m_verbosity(verbosity)
            , m_framemode(framemode)
            , m_randomizationrange(randomizationrange)
            , m_sequencer(sequencer)
            , m_currentSweep(SIZE_MAX)
            , m_currentSequenceId(SIZE_MAX)
            , m_currentFrame(SIZE_MAX)
            , m_epochSize(SIZE_MAX)
        {
            assert(sequencer != nullptr);
            InitializeChunkInformation();
        }

        virtual ~BlockRandomizer()
        {
        }

        size_t getSequenceWindowBegin(size_t sequenceIndex) const
        {
            assert(sequenceIndex < positionchunkwindows.size());
            return positionchunkwindows[sequenceIndex].windowbegin();
        }

        size_t getSequenceWindowEnd(size_t sequenceIndex) const
        {
            assert(sequenceIndex < positionchunkwindows.size());
            return positionchunkwindows[sequenceIndex].windowend();
        }

        virtual void SetEpochConfiguration(const Microsoft::MSR::CNTK::EpochConfiguration& config) override;

        virtual std::vector<Microsoft::MSR::CNTK::InputDescriptionPtr> getInputs() const override
        {
            std::vector<Microsoft::MSR::CNTK::InputDescriptionPtr> dummy;
            return dummy;
        }

        virtual Microsoft::MSR::CNTK::SequenceData getNextSequence() override;
    };
} }
