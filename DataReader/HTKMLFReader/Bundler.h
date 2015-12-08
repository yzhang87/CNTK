//
// <copyright file="utterancesourcemultiNew.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// utterancesourcemultiNew.h -- implementation of utterancesource.h that supports multiple feature and label sets
//

#pragma once

#include "inner_interfaces.h"

#include "Basics.h"                  // for attempt()
#include "htkfeatio.h"                  // for htkmlfreader
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "minibatchsourcehelpers.h"
#include "minibatchiterator.h"
#include "biggrowablevectors.h"
#include "ssematrix.h"
#include "unordered_set"
#include "BlockRandomizer.h"

namespace Microsoft{namespace MSR{namespace CNTK{
    class ConfigParameters;
}}}

namespace msra {
    namespace dbn {

        // ---------------------------------------------------------------------------
        // minibatchutterancesource -- feature source to provide randomized utterances
        // This also implements a frame-wise mode, which is layered on top of the utterance-wise mode
        // and thus benefits from its goodies such as corpus-wide high-level randomization and chunk paging.
        // ---------------------------------------------------------------------------
        class Bundler : public minibatchsource, public Microsoft::MSR::CNTK::Sequencer
        {
            void operator=(const Bundler & other); // non-assignable
            std::vector<size_t> m_vdim;                    // feature dimension after augmenting neighhors
            std::vector<size_t> m_leftcontext;             // number of frames to the left of the target frame in the context window
            std::vector<size_t> m_rightcontext;            // number of frames to the right of the target frame in the context window
            std::vector<unsigned int> m_sampperiod;        // (for reference and to check against model)
            std::vector<string> m_featkind;
            std::vector<size_t> m_featdim;
            const bool m_framemode;           // true -> actually return frame-level randomized frames (not possible in lattice mode)
            std::vector<std::vector<size_t>> m_counts;     // [s] occurrence count for all states (used for priors)
            int m_verbosity;
            // lattice reader
            //const std::vector<unique_ptr<latticesource>> &lattices;
            const latticesource & m_lattices;

            //std::vector<latticesource> lattices;
            // word-level transcripts (for MMI mode when adding best path to lattices)
            const map<wstring, msra::lattices::lattice::htkmlfwordsequence> & m_allwordtranscripts; // (used for getting word-level transcripts)
            //std::vector<map<wstring,msra::lattices::lattice::htkmlfwordsequence>> allwordtranscripts;

            std::vector<std::vector<utterancechunkdata>> m_allchunks;          // set of utterances organized in chunks, referred to by an iterator (not an index)
            std::vector<unique_ptr<biggrowablevector<CLASSIDTYPE>>> m_classids;            // [classidsbegin+t] concatenation of all state sequences
            std::vector<unique_ptr<biggrowablevector<HMMIDTYPE>>> m_phoneboundaries;
            bool issupervised() const { return !m_classids.empty(); }

            size_t m_numutterances;           // total number of utterances
            size_t m_totalframes;            // total frames (same as classids.size() if we have labels)
            double m_timegetbatch;            // [v-hansu] for time measurement
            size_t m_chunksinram;             // (for diagnostics messages)

            // helper to page out a chunk with log message
            void releaserandomizedchunk(size_t k);

            // helper to page in a chunk for a given utterance
            // (window range passed in for checking only)
            // Returns true if we actually did read something.
            bool requirerandomizedchunk(const size_t chunkindex, const size_t windowbegin, const size_t windowend);

            // TODO: this may go away if we store classids directly in the utterance data
            template<class VECTOR> class shiftedvector  // accessing a vector with a non-0 starting index
            {
                void operator= (const shiftedvector &);
                VECTOR & v;
                size_t first;
                size_t n;
                void check(size_t i) const { if (i >= n) LogicError("shiftedvector: index out of bounds"); }
            public:
                shiftedvector(VECTOR & v, size_t first, size_t n) : v(v), first(first), n(n) { }
                // TODO: the following is not templated--do it if needed; also should return a const reference then
                size_t operator[] (size_t i) const { check(i); return v[first + i]; }
            };
            template<class UTTREF> std::vector<shiftedvector<biggrowablevector<CLASSIDTYPE>>> getclassids(const UTTREF & uttref)  // return sub-vector of classids[] for a given utterance
            {
                std::vector<shiftedvector<biggrowablevector<CLASSIDTYPE>>> allclassids;
                allclassids.empty();

                if (!issupervised())
                {
                    foreach_index(i, m_classids)
                        allclassids.push_back(std::move(shiftedvector<biggrowablevector<CLASSIDTYPE>>((*m_classids[i]), 0, 0)));
                    return allclassids;     // nothing to return
                }
                const size_t originalChunkIndex = m_rand->getOriginalChunkIndex(uttref.chunkindex);
                const auto & chunkdata = m_allchunks[0][originalChunkIndex];
                const size_t classidsbegin = chunkdata.getclassidsbegin(uttref.utteranceindex); // index of first state label in global concatenated classids[] array
                const size_t n = chunkdata.numframes(uttref.utteranceindex);
                foreach_index(i, m_classids)
                {
                    if ((*m_classids[i])[classidsbegin + n] != (CLASSIDTYPE)-1)
                        LogicError("getclassids: expected boundary marker not found, internal data structure screwed up");
                    allclassids.push_back(std::move(shiftedvector<biggrowablevector<CLASSIDTYPE>>((*m_classids[i]), classidsbegin, n)));
                }
                return allclassids;   // nothing to return
            }
            template<class UTTREF> std::vector<shiftedvector<biggrowablevector<HMMIDTYPE>>> getphonebound(const UTTREF & uttref)  // return sub-vector of classids[] for a given utterance
            {
                std::vector<shiftedvector<biggrowablevector<HMMIDTYPE>>> allphoneboundaries;
                allphoneboundaries.empty();

                if (!issupervised())
                {
                    foreach_index(i, classids)
                        allphoneboundaries.push_back(std::move(shiftedvector<biggrowablevector<HMMIDTYPE>>((*phoneboundaries[i]), 0, 0)));
                    return allphoneboundaries;     // nothing to return
                }
                const auto & chunk = randomizedchunks[0][uttref.chunkindex];
                const auto & chunkdata = chunk.getchunkdata();
                const size_t classidsbegin = chunkdata.getclassidsbegin(uttref.utteranceindex); // index of first state label in global concatenated classids[] array
                const size_t n = chunkdata.numframes(uttref.utteranceindex);
                foreach_index(i, phoneboundaries)
                {
                    if ((*phoneboundaries[i])[classidsbegin + n] != (HMMIDTYPE)-1)
                        LogicError("getclassids: expected boundary marker not found, internal data structure screwed up");
                    allphoneboundaries.push_back(std::move(shiftedvector<biggrowablevector<HMMIDTYPE>>((*phoneboundaries[i]), classidsbegin, n)));
                }
                return allphoneboundaries;   // nothing to return
            }

        private:
            class matrixasvectorofvectors  // wrapper around a matrix that views it as a vector of column vectors
            {
                void operator= (const matrixasvectorofvectors &);  // non-assignable
                msra::dbn::matrixbase & m;
            public:
                matrixasvectorofvectors(msra::dbn::matrixbase & m) : m(m) {}
                size_t size() const { return m.cols(); }
                const_array_ref<float> operator[] (size_t j) const { return array_ref<float>(&m(0, j), m.rows()); }
            };


        public:
            Bundler::Bundler(
                const Microsoft::MSR::CNTK::ConfigParameters & readerConfig,
                const std::vector<std::vector<wstring>> & infiles,
                const std::vector<map<wstring, std::vector<msra::asr::htkmlfentry>>> & labels,
                std::vector<size_t> vdim,
                std::vector<size_t> udim,
                std::vector<size_t> leftcontext,
                std::vector<size_t> rightcontext,
                size_t randomizationrange,
                const latticesource & lattices,
                const map<wstring, msra::lattices::lattice::htkmlfwordsequence> & allwordtranscripts,
                const bool framemode,
                std::vector<Microsoft::MSR::CNTK::FrameDescription> featureFrameDescriptions,
                std::vector<Microsoft::MSR::CNTK::FrameDescription> labelFrameDescriptions,
                std::vector<Microsoft::MSR::CNTK::InputDescriptionPtr> inputs,
                std::map<std::wstring, size_t> nameToId,
                std::map<std::wstring, size_t> featureNameToIdMap,
                std::map<std::wstring, size_t> labelNameToIdMap,
                size_t elementSize);

            // eldak: Should go away.
            void SetRandomizer(std::shared_ptr<BlockRandomizer> rand)
            {
                m_rand = rand;
            }

            virtual void SetEpochConfiguration(const Microsoft::MSR::CNTK::EpochConfiguration& config);

            void setverbosity(int newverbosity){ m_verbosity = newverbosity; }

            // get the next minibatch
            // A minibatch is made up of one or more utterances.
            // We will return less than 'framesrequested' unless the first utterance is too long.
            // Note that this may return frames that are beyond the epoch end, but the first frame is always within the epoch.
            // We specify the utterance by its global start time (in a space of a infinitely repeated training set).
            // This is efficient since getbatch() is called with sequential 'globalts' except at epoch start.
            // Note that the start of an epoch does not necessarily fall onto an utterance boundary. The caller must use firstvalidglobalts() to find the first valid globalts at or after a given time.
            // Support for data parallelism:  If mpinodes > 1 then we will
            //  - load only a subset of blocks from the disk
            //  - skip frames/utterances in not-loaded blocks in the returned data
            //  - 'framesadvanced' will still return the logical #frames; that is, by how much the global time index is advanced
            void getbatch(const size_t globalts, const size_t framesrequested,
                const size_t subsetnum, const size_t numsubsets, size_t & framesadvanced,
                std::vector<msra::dbn::matrix> & feat, std::vector<std::vector<size_t>> & uids,
                std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> & transcripts,
                std::vector<shared_ptr<const latticesource::latticepair>> & latticepairs, std::vector<std::vector<size_t>> & sentendmark,
                std::vector<std::vector<size_t>> & phoneboundaries) override;

            bool supportsbatchsubsetting() const override
            {
                return true;
            }

            void getbatch(const size_t globalts,
                const size_t framesrequested, std::vector<msra::dbn::matrix> & feat, std::vector<std::vector<size_t>> & uids,
                std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> & transcripts,
                std::vector<shared_ptr<const latticesource::latticepair>> & lattices, std::vector<std::vector<size_t>> & sentendmark,
                std::vector<std::vector<size_t>> & phoneboundaries);

            double gettimegetbatch() { return m_timegetbatch; }

            // alternate (updated) definition for multiple inputs/outputs - read as a vector of feature matrixes or a vector of label strings
            void getbatch(const size_t /*globalts*/,
                const size_t /*framesrequested*/, msra::dbn::matrix & /*feat*/, std::vector<size_t> & /*uids*/,
                std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> & /*transcripts*/,
                std::vector<shared_ptr<const latticesource::latticepair>> & /*latticepairs*/);

            size_t totalframes() const { return m_totalframes; }

            // return first valid globalts to ask getbatch() for
            // In utterance mode, the epoch start may fall in the middle of an utterance.
            // We return the end time of that utterance (which, in pathological cases, may in turn be outside the epoch; handle that).
            /*implement*/ size_t firstvalidglobalts(const size_t globalts); // TODO can be const

            const std::vector<size_t> & unitcounts() const { return m_counts[0]; }

            virtual const Microsoft::MSR::CNTK::Timeline& getTimeline() const override;

            virtual std::vector<Microsoft::MSR::CNTK::InputDescriptionPtr> getInputs() const override;

            virtual Microsoft::MSR::CNTK::SequenceData getSequenceById(size_t id) override;
            bool RequireChunk(size_t chunkindex);
            void ReleaseChunk(size_t chunkIndex);

            private:

            Microsoft::MSR::CNTK::Timeline m_timeline;
            std::map<size_t, const utterancedesc*> m_sequenceIdToSequence;
            size_t m_workerRank;
            size_t m_numberOfWorkers;
            size_t m_epochSize;
            //size_t m_currentSampleCount;
            size_t m_elementSize;
            std::vector<size_t> m_udim;
            std::vector<Microsoft::MSR::CNTK::FrameDescription> m_featureFrameDescriptions;
            std::vector<Microsoft::MSR::CNTK::FrameDescription> m_labelFrameDescriptions;
            std::vector<Microsoft::MSR::CNTK::InputDescriptionPtr> m_inputs;
            std::map<std::wstring, size_t> m_nameToId;
            std::map<std::wstring, size_t> m_featureNameToIdMap;
            std::map<std::wstring, size_t> m_labelNameToIdMap;

            std::vector<BlockRandomizer::sequenceref> m_sequences;

            // return sub-vector of classids[] for a given utterance
            std::vector<shiftedvector<biggrowablevector<CLASSIDTYPE>>> GetClassIds(
                const BlockRandomizer::sequenceref& uttref)
            {
                std::vector<shiftedvector<biggrowablevector<CLASSIDTYPE>>> allclassids;
                allclassids.empty();

                if (!issupervised())
                {
                    foreach_index(i, m_classids)
                        allclassids.push_back(std::move(shiftedvector<biggrowablevector<CLASSIDTYPE>>((*m_classids[i]), 0, 0)));
                    return allclassids;     // nothing to return
                }
                const size_t originalChunkIndex = uttref.chunkindex;
                const auto & chunkdata = m_allchunks[0][originalChunkIndex];
                const size_t classidsbegin = chunkdata.getclassidsbegin(uttref.utteranceindex); // index of first state label in global concatenated classids[] array
                const size_t n = chunkdata.numframes(uttref.utteranceindex);
                foreach_index(i, m_classids)
                {
                    if ((*m_classids[i])[classidsbegin + n] != (CLASSIDTYPE)-1)
                        LogicError("getclassids: expected boundary marker not found, internal data structure screwed up");
                    allclassids.push_back(std::move(shiftedvector<biggrowablevector<CLASSIDTYPE>>((*m_classids[i]), classidsbegin, n)));
                }
                return allclassids;   // nothing to return
            }

            std::shared_ptr<BlockRandomizer> m_rand;
            bool m_partialMinibatch;
        };
    }
}
