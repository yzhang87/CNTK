//
// <copyright file="BundlerSplitted.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "InnerInterfaces.h"

#include "Basics.h"                  // for attempt()
#include "htkfeatio.h"                  // for htkmlfreader
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "minibatchsourcehelpers.h"
#include "minibatchiterator.h"
#include "biggrowablevectors.h"
#include "ssematrix.h"
#include "BlockRandomizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ConfigParameters;

    // data store (incl. paging in/out of features and lattices)
    struct utterancedesc            // data descriptor for one utterance
    {
        msra::asr::htkfeatreader::parsedpath parsedpath;    // archive filename and frame range in that file
        size_t classidsbegin;       // index into allclassids[] array (first frame)

        utterancedesc(msra::asr::htkfeatreader::parsedpath&& ppath, size_t classidsbegin) : parsedpath(std::move(ppath)), classidsbegin(classidsbegin) {}

        const wstring & logicalpath() const { return parsedpath; /*type cast will return logical path*/ }
        size_t numframes() const { return parsedpath.numframes(); }
        wstring key() const                           // key used for looking up lattice (not stored to save space)
        {
#ifdef _MSC_VER
            static const wstring emptywstring;
            static const wregex deleteextensionre(L"\\.[^\\.\\\\/:]*$");
            return regex_replace(logicalpath(), deleteextensionre, emptywstring);  // delete extension (or not if none)
#else
            return removeExtension(logicalpath());
#endif
        }
    };

    // Make sure type 'utterancedesc' has a move constructor
    static_assert(std::is_move_constructible<utterancedesc>::value, "Type 'utterancedesc' should be move constructible!");

    struct utterancechunkdata       // data for a chunk of utterances
    {
        std::vector<utterancedesc> utteranceset;    // utterances in this set
        size_t numutterances() const { return utteranceset.size(); }

        std::vector<size_t> firstframes;    // [utteranceindex] first frame for given utterance
        mutable msra::dbn::matrix frames;   // stores all frames consecutively (mutable since this is a cache)
        size_t totalframes;         // total #frames for all utterances in this chunk
        mutable std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>> lattices;   // (may be empty if none)

        // construction
        utterancechunkdata() : totalframes(0) {}
        void push_back(utterancedesc &&/*destructive*/ utt)
        {
            if (isinram())
                LogicError("utterancechunkdata: frames already paged into RAM--too late to add data");
            firstframes.push_back(totalframes);
            totalframes += utt.numframes();
            utteranceset.push_back(std::move(utt));
        }

        // accessors to an utterance's data
        size_t numframes(size_t i) const { return utteranceset[i].numframes(); }
        size_t getclassidsbegin(size_t i) const { return utteranceset[i].classidsbegin; }
        msra::dbn::matrixstripe getutteranceframes(size_t i) const // return the frame set for a given utterance
        {
            if (!isinram())
                LogicError("getutteranceframes: called when data have not been paged in");
            const size_t ts = firstframes[i];
            const size_t n = numframes(i);
            return msra::dbn::matrixstripe(frames, ts, n);
        }
        shared_ptr<const msra::dbn::latticesource::latticepair> getutterancelattice(size_t i) const // return the frame set for a given utterance
        {
            if (!isinram())
                LogicError("getutterancelattice: called when data have not been paged in");
            return lattices[i];
        }

        // paging
        // test if data is in memory at the moment
        bool isinram() const { return !frames.empty(); }
        // page in data for this chunk
        // We pass in the feature info variables by ref which will be filled lazily upon first read
        void requiredata(string & featkind, size_t & featdim, unsigned int & sampperiod, const msra::dbn::latticesource & latticesource, int verbosity = 0) const
        {
            if (numutterances() == 0)
                LogicError("requiredata: cannot page in virgin block");
            if (isinram())
                LogicError("requiredata: called when data is already in memory");
            try             // this function supports retrying since we read from the unrealible network, i.e. do not return in a broken state
            {
                msra::asr::htkfeatreader reader;    // feature reader (we reinstantiate it for each block, i.e. we reopen the file actually)
                // if this is the first feature read ever, we explicitly open the first file to get the information such as feature dimension
                if (featdim == 0)
                {
                    reader.getinfo(utteranceset[0].parsedpath, featkind, featdim, sampperiod);
                    fprintf(stderr, "requiredata: determined feature kind as %llu-dimensional '%s' with frame shift %.1f ms\n",
                        featdim, featkind.c_str(), sampperiod / 1e4);
                }
                // read all utterances; if they are in the same archive, htkfeatreader will be efficient in not closing the file
                frames.resize(featdim, totalframes);
                if (!latticesource.empty())
                    lattices.resize(utteranceset.size());
                foreach_index(i, utteranceset)
                {
                    //fprintf (stderr, ".");
                    // read features for this file
                    auto uttframes = getutteranceframes(i);    // matrix stripe for this utterance (currently unfilled)
                    reader.read(utteranceset[i].parsedpath, (const string &)featkind, sampperiod, uttframes);  // note: file info here used for checkuing only
                    // page in lattice data
                    if (!latticesource.empty())
                        latticesource.getlattices(utteranceset[i].key(), lattices[i], uttframes.cols());
                }
                //fprintf (stderr, "\n");
                if (verbosity)
                    fprintf(stderr, "requiredata: %d utterances read\n", (int)utteranceset.size());
            }
            catch (...)
            {
                releasedata();
                throw;
            }
        }
        // page out data for this chunk
        void releasedata() const
        {
            if (numutterances() == 0)
                LogicError("releasedata: cannot page out virgin block");
            if (!isinram())
                LogicError("releasedata: called when data is not memory");
            // release frames
            frames.resize(0, 0);
            // release lattice data
            lattices.clear();
        }
    };

    // ---------------------------------------------------------------------------
    // minibatchutterancesource -- feature source to provide randomized utterances
    // This also implements a frame-wise mode, which is layered on top of the utterance-wise mode
    // and thus benefits from its goodies such as corpus-wide high-level randomization and chunk paging.
    // ---------------------------------------------------------------------------
    class BundlerSplitted : public Sequencer
    {
        std::vector<size_t> m_featureIndices;
        std::vector<size_t> m_labelIndices;

        void operator=(const BundlerSplitted & other); // non-assignable

        std::vector<size_t> m_leftcontext;             // number of frames to the left of the target frame in the context window
        std::vector<size_t> m_rightcontext;            // number of frames to the right of the target frame in the context window
        std::vector<size_t> m_featdim;
        const bool m_framemode;           // true -> actually return frame-level randomized frames (not possible in lattice mode)
        int m_verbosity;

        std::vector<std::vector<utterancechunkdata>> m_allchunks;          // set of utterances organized in chunks, referred to by an iterator (not an index)
        std::vector<unique_ptr<msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE>>> m_classids;            // [classidsbegin+t] concatenation of all state sequences

        bool issupervised() const { return !m_classids.empty(); }

        size_t m_totalframes;            // total frames (same as classids.size() if we have labels)
        double m_timegetbatch;            // [v-hansu] for time measurement
        size_t m_chunksinram;             // (for diagnostics messages)

        // helper to page out a chunk with log message
        //void releaserandomizedchunk(size_t k);

        // helper to page in a chunk for a given utterance
        // (window range passed in for checking only)
        // Returns true if we actually did read something.
        //bool requirerandomizedchunk(const size_t chunkindex, const size_t windowbegin, const size_t windowend);

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
        BundlerSplitted::BundlerSplitted(const ConfigParameters& readerConfig, bool framemode, size_t elementSize);

        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;

        virtual const Timeline& GetTimeline() const override;
        virtual std::vector<InputDescriptionPtr> GetInputs() const override;
        virtual SequenceData GetSequenceById(size_t id) override;
        virtual bool RequireChunk(size_t chunkindex) override;
        virtual void ReleaseChunk(size_t chunkIndex) override;

    private:

        Timeline m_timeline;
        std::map<size_t, const utterancedesc*> m_sequenceIdToSequence;
        size_t m_workerRank;
        size_t m_numberOfWorkers;
        size_t m_elementSize;
        //std::vector<size_t> m_udim;
        std::vector<FrameDescription> m_featureFrameDescriptions;
        std::vector<FrameDescription> m_labelFrameDescriptions;
        std::vector<InputDescriptionPtr> m_inputs;

        // TODO can more stuff be dropped?
        struct sequenceref              // described a sequence to be randomized (in frame mode, a single frame; a full utterance otherwise)
        {
            size_t chunkindex;          // lives in this chunk (index into randomizedchunks[])
            size_t utteranceindex;      // utterance index in that chunk
            size_t numframes;           // (cached since we cannot directly access the underlying data from here)
            size_t frameindex;          // 0 for utterances

            sequenceref()
                : chunkindex(0)
                , utteranceindex(0)
                , frameindex(0)
                , numframes(0)
            {}
            sequenceref(size_t chunkindex, size_t utteranceindex, size_t frameindex = 0)
                : chunkindex(chunkindex)
                , utteranceindex(utteranceindex)
                , frameindex(frameindex)
                , numframes(0)
            {}
        };

        std::vector<sequenceref> m_sequences;

        // return sub-vector of classids[] for a given utterance
        std::vector<shiftedvector<msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE>>> GetClassIds(
            const sequenceref& uttref)
        {
            std::vector<shiftedvector<msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE>>> allclassids;
            allclassids.empty();

            if (!issupervised())
            {
                foreach_index(i, m_classids)
                    allclassids.push_back(std::move(shiftedvector<msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE>>((*m_classids[i]), 0, 0)));
                return allclassids;     // nothing to return
            }
            const size_t originalChunkIndex = uttref.chunkindex;
            const auto & chunkdata = m_allchunks[0][originalChunkIndex];
            const size_t classidsbegin = chunkdata.getclassidsbegin(uttref.utteranceindex); // index of first state label in global concatenated classids[] array
            const size_t n = chunkdata.numframes(uttref.utteranceindex);
            foreach_index(i, m_classids)
            {
                if ((*m_classids[i])[classidsbegin + n] != (msra::dbn::CLASSIDTYPE) - 1)
                {
                    LogicError("getclassids: expected boundary marker not found, internal data structure screwed up");
                }
                allclassids.push_back(std::move(shiftedvector<msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE>>((*m_classids[i]), classidsbegin, n)));
            }
            return allclassids;   // nothing to return
        }
    };
}}}
