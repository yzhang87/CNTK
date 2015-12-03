//
// <copyright file="utterancesourcemultiNew.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// utterancesourcemultiNew.h -- implementation of utterancesource.h that supports multiple feature and label sets
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
#include "BlockRandomizer.h"

namespace msra {
    namespace dbn {

        // ---------------------------------------------------------------------------
        // minibatchutterancesource -- feature source to provide randomized utterances
        // This also implements a frame-wise mode, which is layered on top of the utterance-wise mode
        // and thus benefits from its goodies such as corpus-wide high-level randomization and chunk paging.
        // ---------------------------------------------------------------------------
        class Bundler : public minibatchsource
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


            std::unique_ptr<BlockRandomizer> rand;

            // helper to page out a chunk with log message
            void releaserandomizedchunk(size_t k)
            {
                size_t numreleased = 0;
                size_t numStreams = m_allchunks.size();
                for (size_t m = 0; m < numStreams; m++)
                {
                    auto & chunkdata = rand->getChunkData(m, k);
                    if (chunkdata.isinram())
                    {
#if 0 // TODO restore diagnostics
                        if (verbosity)
                            fprintf(stderr, "releaserandomizedchunk: paging out randomized chunk %u (frame range [%d..%d]), %d resident in RAM\n",
                            (int)k, (int)randomizedchunks[m][k].globalts, (int)(randomizedchunks[m][k].globalte() - 1), (int)(chunksinram - 1));
#endif
                        chunkdata.releasedata();
                        numreleased++;
                    }
                }
                if (numreleased > 0 && numreleased < numStreams)
                {
                    LogicError("releaserandomizedchunk: inconsistency detected - some inputs have chunks in ram, some not");
                }
                else if (numreleased == numStreams)
                {
                    m_chunksinram--;
                }
                return;
            }

            // helper to page in a chunk for a given utterance
            // (window range passed in for checking only)
            // Returns true if we actually did read something.
            bool requirerandomizedchunk(const size_t chunkindex, const size_t windowbegin, const size_t windowend)
            {
                size_t numinram = 0;

                if (chunkindex < windowbegin || chunkindex >= windowend)
                    LogicError("requirerandomizedchunk: requested utterance outside in-memory chunk range");

                size_t numStreams = m_allchunks.size();
                for (size_t m = 0; m < numStreams; m++)
                {
                    auto & chunkdata = rand->getChunkData(m, chunkindex);
                    if (chunkdata.isinram())
                        numinram++;
                }
                if (numinram == numStreams)
                    return false;
                else if (numinram == 0)
                {
                    for (size_t m = 0; m < numStreams; m++)
                    {
                        auto & chunkdata = rand->getChunkData(m, chunkindex);
#if 0 // TODO restore diagnostics
                        if (verbosity)
                            fprintf(stderr, "feature set %u: requirerandomizedchunk: paging in randomized chunk %llu (frame range [%llu..%llu]), %llu resident in RAM\n",
                            m, chunkindex, chunk.globalts, (chunk.globalte() - 1), (chunksinram + 1));
#endif
                        msra::util::attempt(5, [&]()   // (reading from network)
                        {
                            chunkdata.requiredata(m_featkind[m], m_featdim[m], m_sampperiod[m], m_lattices, m_verbosity);
                        });
                    }
                    m_chunksinram++;
                    return true;
                }
                else
                {
                    LogicError("requirerandomizedchunk: inconsistency detected - some inputs need chunks paged in, some not");
                }
            }

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
                const auto & chunkdata = rand->getChunkData(0, uttref.chunkindex);
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

        public:

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
            Bundler(
                const std::vector<std::vector<wstring>> & infiles,
                const std::vector<map<wstring, std::vector<msra::asr::htkmlfentry>>> & labels,
                std::vector<size_t> vdim,
                std::vector<size_t> udim,
                std::vector<size_t> leftcontext,
                std::vector<size_t> rightcontext,
                size_t randomizationrange,
                const latticesource & lattices,
                const map<wstring, msra::lattices::lattice::htkmlfwordsequence> & allwordtranscripts,
                const bool framemode);

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
                std::vector<std::vector<size_t>> & phoneboundaries) override
            {
                bool readfromdisk = false;  // return value: shall be 'true' if we paged in anything

                auto_timer timergetbatch;
                assert(m_totalframes > 0);

                // update randomization if a new sweep is entered  --this is a complex operation that updates many of the data members used below
                const size_t sweep = rand->lazyrandomization(globalts, m_allchunks);

                size_t mbframes = 0;
                const std::vector<char> noboundaryflags;    // dummy

                sentendmark;
                phoneboundaries;
#undef EXPERIMENTAL_UNIFIED_PATH
#ifdef EXPERIMENTAL_UNIFIED_PATH
                // find utterance position for globalts
                // There must be a precise match; it is not possible to specify frames that are not on boundaries.
                auto positer = randomizedutteranceposmap.find(globalts);
                if (positer == randomizedutteranceposmap.end())
                    LogicError("getbatch: invalid 'globalts' parameter; must match an existing utterance boundary");
                const size_t spos = positer->second;

                size_t numsequences = framemode ? _totalframes : numutterances;

                // determine how many utterances will fit into the requested minibatch size
                mbframes = randomizedutterancerefs[spos].numframes;   // at least one utterance, even if too long
                size_t epos;
                for (epos = spos + 1; epos < numsequences /* numutterances */ && ((mbframes + randomizedutterancerefs[epos].numframes) < framesrequested); epos++)  // add more utterances as long as they fit within requested minibatch size
                    mbframes += randomizedutterancerefs[epos].numframes;

                // do some paging housekeeping
                // This will also set the feature-kind information if it's the first time.
                // Free all chunks left of the range.
                // Page-in all chunks right of the range.
                // We are a little more blunt for now: Free all outside the range, and page in only what is touched. We could save some loop iterations.
                const size_t windowbegin = positionchunkwindows[spos].windowbegin();
                const size_t windowend = positionchunkwindows[epos - 1].windowend();
                for (size_t k = 0; k < windowbegin; k++)
                    releaserandomizedchunk(k);
                for (size_t k = windowend; k < randomizedchunks[0].size(); k++)
                    releaserandomizedchunk(k);
                for (size_t pos = spos; pos < epos; pos++)
                    if ((randomizedutterancerefs[pos].chunkindex % numsubsets) == subsetnum)
                        readfromdisk |= requirerandomizedchunk(randomizedutterancerefs[pos].chunkindex, windowbegin, windowend); // (window range passed in for checking only)

                // Note that the above loop loops over all chunks incl. those that we already should have.
                // This has an effect, e.g., if 'numsubsets' has changed (we will fill gaps).

                // determine the true #frames we return, for allocation--it is less than mbframes in the case of MPI/data-parallel sub-set mode
                size_t tspos = 0;
                for (size_t pos = spos; pos < epos; pos++)
                {
                    const auto & uttref = randomizedutterancerefs[pos];
                    if ((uttref.chunkindex % numsubsets) != subsetnum)            // chunk not to be returned for this MPI node
                        continue;

                    tspos += uttref.numframes;
                }

                // resize feat and uids
                feat.resize(vdim.size());
                uids.resize(classids.size());
                phoneboundaries.resize(classids.size());
                sentendmark.resize(vdim.size());
                assert(feat.size() == vdim.size());
                assert(feat.size() == randomizedchunks.size());

                // TODO should still work for !framemode; for framemode more work is needed:
                // - subsetsizes computation - augmentation still crashes
                foreach_index(i, feat)
                {
                    feat[i].resize(vdim[i], tspos /* TODO versus allocframes */);

                    if (i == 0)
                    {
                        foreach_index(j, uids)
                        {
                            if (issupervised())             // empty means unsupervised training -> return empty uids
                            {
                                uids[j].resize(tspos);
                                phoneboundaries[j].resize(tspos);
                            }
                            else
                            {
                                uids[i].clear();
                                phoneboundaries[i].clear();
                            }
                            latticepairs.clear();               // will push_back() below
                            transcripts.clear();
                        }
                        foreach_index(j, sentendmark)
                        {
                            sentendmark[j].clear();
                        }
                    }
                }

                if (verbosity > 0)
                    fprintf(stderr, "getbatch: getting utterances %lu..%lu (%lu subset of %lu frames out of %lu requested) in sweep %lu\n",
                    spos, (epos - 1), tspos, mbframes, framesrequested, sweep);
                tspos = 0;   // relative start of utterance 'pos' within the returned minibatch
                for (size_t pos = spos; pos < epos; pos++)
                {
                    const auto & uttref = randomizedutterancerefs[pos];
                    if ((uttref.chunkindex % numsubsets) != subsetnum)            // chunk not to be returned for this MPI node
                        continue;

                    size_t n = 0;
                    foreach_index(i, randomizedchunks)
                    {
                        const auto & chunk = randomizedchunks[i][uttref.chunkindex];
                        const auto & chunkdata = chunk.getchunkdata();
                        assert((numsubsets > 1) || (uttref.globalts == globalts + tspos));
                        auto uttframes = chunkdata.getutteranceframes(uttref.utteranceindex);
                        matrixasvectorofvectors uttframevectors(uttframes);    // (wrapper that allows m[j].size() and m[j][i] as required by augmentneighbors())
                        n = uttref.numframes;
                        const size_t uttNumFramesFromVector = uttframevectors.size();
                        sentendmark[i].push_back(n + tspos);
                        // TODO rejoin
                        assert(uttNumFramesFromVector == uttframes.cols()); uttNumFramesFromVector;
                        assert(n == (framemode ? 1 : uttNumFramesFromVector));
                        assert(chunkdata.numframes(uttref.utteranceindex) == uttNumFramesFromVector);

                        size_t frameIndex = uttref.frameindex;

                        // copy the frames and class labels
                        for (size_t t = 0; t < n; t++, frameIndex++)          // t = time index into source utterance
                        {
                            size_t leftextent, rightextent;
                            // page in the needed range of frames
                            // TODO hoist?
                            if (leftcontext[i] == 0 && rightcontext[i] == 0)
                            {
                                leftextent = rightextent = augmentationextent(uttframevectors[frameIndex].size(), vdim[i]);
                            }
                            else
                            {
                                leftextent = leftcontext[i];
                                rightextent = rightcontext[i];
                            }

                            // TODO memory-safe, maybe not correct
                            augmentneighbors(uttframevectors, noboundaryflags, frameIndex, leftextent, rightextent, feat[i], t /* frameIndex */ + tspos);
                            //augmentneighbors(uttframevectors, noboundaryflags, frameIndex, leftextent, rightextent, feat[i], currmpinodeframecount);
                        }

                        // copy the frames and class labels
                        if (i == 0)
                        {
                            auto uttclassids = getclassids(uttref);
                            auto uttphoneboundaries = getphonebound(uttref);
                            foreach_index(j, uttclassids)
                            {
                                for (size_t t = 0; t < n; t++)          // t = time index into source utterance
                                {
                                    if (issupervised())
                                    {
                                        uids[j][t + tspos] = uttclassids[j][t];
                                        phoneboundaries[j][t + tspos] = uttphoneboundaries[j][t];
                                    }
                                }

                                if (!this->lattices.empty())
                                {
                                    auto latticepair = chunkdata.getutterancelattice(uttref.utteranceindex);
                                    latticepairs.push_back(latticepair);
                                    // look up reference
                                    const auto & key = latticepair->getkey();
                                    if (!allwordtranscripts.empty())
                                    {
                                        const auto & transcript = allwordtranscripts.find(key)->second;
                                        transcripts.push_back(transcript.words);
                                    }
                                }
                            }
                        }
                    }
                    tspos += n;
                }

                foreach_index(i, feat)
                {
                    assert(tspos == feat[i].cols());
                }
#endif

                if (!m_framemode)      // regular utterance mode
                {
                    assert(0); // looking at frame-mode scenario for now
                    // TODO code was moved up
                }
                else
                {
                    const size_t sweepts = sweep * m_totalframes;         // first global frame index for this sweep
                    const size_t sweepte = sweepts + m_totalframes;       // and its end
                    const size_t globalte = min(globalts + framesrequested, sweepte);  // we return as much as requested, but not exceeding sweep end
                    mbframes = globalte - globalts;        // that's our mb size

                    // determine window range
                    // We enumerate all frames--can this be done more efficiently?
                    const size_t firstchunk = rand->chunkForFramePos(globalts);
                    const size_t lastchunk = rand->chunkForFramePos(globalte - 1);

                    assert(lastchunk <= firstchunk + 1); // shouldn't really cover more than two chunks...?
                    const size_t windowbegin = rand->getChunkWindowBegin(firstchunk);
                    const size_t windowend = rand->getChunkWindowEnd(lastchunk);
                    const size_t numChunks = m_allchunks[0].size();
                    const size_t numStreams = m_allchunks.size();
                    if (m_verbosity > 0)
                        fprintf(stderr, "getbatch: getting randomized frames [%d..%d] (%d frames out of %d requested) in sweep %d; chunks [%d..%d] -> chunk window [%d..%d)\n",
                        (int)globalts, (int)globalte, (int)mbframes, (int)framesrequested, (int)sweep, (int)firstchunk, (int)lastchunk, (int)windowbegin, (int)windowend);
                    // release all data outside, and page in all data inside
                    for (size_t k = 0; k < windowbegin; k++)
                        releaserandomizedchunk(k);
                    for (size_t k = windowbegin; k < windowend; k++)
                        if ((k % numsubsets) == subsetnum)        // in MPI mode, we skip chunks this way
                            readfromdisk |= requirerandomizedchunk(k, windowbegin, windowend); // (window range passed in for checking only, redundant here)
                    for (size_t k = windowend; k < numChunks; k++)
                        releaserandomizedchunk(k);

                    // determine the true #frames we return--it is less than mbframes in the case of MPI/data-parallel sub-set mode
                    // First determine it for all nodes, then pick the min over all nodes, as to give all the same #frames for better load balancing.
                    // TODO: No, return all; and leave it to caller to redistribute them [Zhijie Yan]
                    std::vector<size_t> subsetsizes(numsubsets, 0);
                    for (size_t i = 0; i < mbframes; i++)   // i is input frame index; j < i in case of MPI/data-parallel sub-set mode
                    {
                        const size_t framepos = (globalts + i) % m_totalframes;  // (for comments, see main loop below)
                        //const sequenceref & frameref = randomizedframerefs[framepos];
                        const BlockRandomizer::sequenceref & frameref = rand->getSequenceRef(framepos);
                        subsetsizes[frameref.chunkindex % numsubsets]++;
                    }
                    size_t j = subsetsizes[subsetnum];        // return what we have  --TODO: we can remove the above full computation again now
                    const size_t allocframes = max(j, (mbframes + numsubsets - 1) / numsubsets);  // we leave space for the desired #frames, assuming caller will try to pad them later

                    // resize feat and uids
                    feat.resize(m_vdim.size());
                    uids.resize(m_classids.size());
                    assert(feat.size() == m_vdim.size());
                    assert(feat.size() == numStreams);
                    foreach_index(i, feat)
                    {
                        feat[i].resize(m_vdim[i], allocframes);
                        feat[i].shrink(m_vdim[i], j);
                    }

                    foreach_index(k, uids)
                    {
                        if (issupervised())             // empty means unsupervised training -> return empty uids
                            uids[k].resize(j);
                        else
                            uids[k].clear();
                        latticepairs.clear();               // will push_back() below
                        transcripts.clear();
                    }

                    // return randomized frames for the time range of those utterances
                    size_t currmpinodeframecount = 0;
                    for (size_t j = 0; j < mbframes; j++)
                    {
                        if (currmpinodeframecount >= feat[0].cols())               // MPI/data-parallel mode: all nodes return the same #frames, which is how feat(,) is allocated
                            break;

                        // map to time index inside arrays
                        const size_t framepos = (globalts + j) % m_totalframes;  // using mod because we may actually run beyond the sweep for the last call
                        //const sequenceref & frameref = randomizedframerefs[framepos];
                        const BlockRandomizer::sequenceref & frameref = rand->getSequenceRef(framepos);

                        // in MPI/data-parallel mode, skip frames that are not in chunks loaded for this MPI node
                        if ((frameref.chunkindex % numsubsets) != subsetnum)
                            continue;

                        // random utterance
                        readfromdisk |= requirerandomizedchunk(frameref.chunkindex, windowbegin, windowend);    // (this is just a check; should not actually page in anything)

                        for (size_t i = 0; i < numStreams; i++)
                        {
                            const auto & chunkdata = rand->getChunkData(i, frameref.chunkindex);
                            auto uttframes = chunkdata.getutteranceframes(frameref.utteranceindex);
                            matrixasvectorofvectors uttframevectors(uttframes);    // (wrapper that allows m[.].size() and m[.][.] as required by augmentneighbors())
                            const size_t n = uttframevectors.size();
                            assert(n == uttframes.cols() && chunkdata.numframes(frameref.utteranceindex) == n); n;

                            // copy frame and class labels
                            const size_t t = frameref.frameindex;

                            size_t leftextent, rightextent;
                            // page in the needed range of frames
                            if (m_leftcontext[i] == 0 && m_rightcontext[i] == 0)
                            {
                                leftextent = rightextent = augmentationextent(uttframevectors[t].size(), m_vdim[i]);
                            }
                            else
                            {
                                leftextent = m_leftcontext[i];
                                rightextent = m_rightcontext[i];
                            }
                            augmentneighbors(uttframevectors, noboundaryflags, t, leftextent, rightextent, feat[i], currmpinodeframecount);

                            if (issupervised() && i == 0)
                            {
                                auto frameclassids = getclassids(frameref);
                                foreach_index(k, uids)
                                    uids[k][currmpinodeframecount] = frameclassids[k][t];
                            }
                        }

                        currmpinodeframecount++;
                    }
                }
                m_timegetbatch = timergetbatch;

                // this is the number of frames we actually moved ahead in time
                framesadvanced = mbframes;
            }

            bool supportsbatchsubsetting() const override
            {
                return true;
            }

            void getbatch(const size_t globalts,
                const size_t framesrequested, std::vector<msra::dbn::matrix> & feat, std::vector<std::vector<size_t>> & uids,
                std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> & transcripts,
                std::vector<shared_ptr<const latticesource::latticepair>> & lattices, std::vector<std::vector<size_t>> & sentendmark,
                std::vector<std::vector<size_t>> & phoneboundaries)
            {
                size_t dummy;
                getbatch(globalts, framesrequested, 0, 1, dummy, feat, uids, transcripts, lattices, sentendmark, phoneboundaries);
            }

            double gettimegetbatch() { return m_timegetbatch; }

            // alternate (updated) definition for multiple inputs/outputs - read as a vector of feature matrixes or a vector of label strings
            void getbatch(const size_t /*globalts*/,
                const size_t /*framesrequested*/, msra::dbn::matrix & /*feat*/, std::vector<size_t> & /*uids*/,
                std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> & /*transcripts*/,
                std::vector<shared_ptr<const latticesource::latticepair>> & /*latticepairs*/)
            {
                // should never get here
                RuntimeError("minibatchframesourcemulti: getbatch() being called for single input feature and single output feature, should use minibatchutterancesource instead\n");

                // for single input/output set size to be 1 and run old getbatch
                //feat.resize(1);
                //uids.resize(1);
                //return getbatch(globalts, framesrequested, feat[0], uids[0], transcripts, latticepairs);
            }

            size_t totalframes() const { return m_totalframes; }

            // return first valid globalts to ask getbatch() for
            // In utterance mode, the epoch start may fall in the middle of an utterance.
            // We return the end time of that utterance (which, in pathological cases, may in turn be outside the epoch; handle that).
            /*implement*/ size_t firstvalidglobalts(const size_t globalts) // TODO can be const
            {
                // update randomization if a new sweep is entered
                const size_t sweep = rand->lazyrandomization(globalts, m_allchunks);

                // frame mode: start at sweep boundary directly // TODO so globalts needs to be at sweep boundary?
                if (m_framemode)
                    return globalts;
                // utterance mode
                assert(globalts >= sweep * m_totalframes && globalts < (sweep + 1) * m_totalframes); sweep;
                // TODO use std::find
                size_t pos;
                for (pos = 0; pos < rand->getNumSequences(); pos++)
                    if (rand->getSequenceRef(pos).globalts >= globalts)
                        return rand->getSequenceRef(pos).globalts;   // exact or inexact match
                return rand->getSequenceRef(pos - 1).globalte();     // boundary case: requested time falls within the last utterance
            }

            const std::vector<size_t> & unitcounts() const { return m_counts[0]; }
        };

    }
}
