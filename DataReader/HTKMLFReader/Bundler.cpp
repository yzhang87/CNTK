#include "stdafx.h"
#include "Bundler.h"
#include <DataReader.h>

using namespace Microsoft::MSR::CNTK;

namespace msra { namespace dbn {

    // constructor
    // Pass empty labels to denote unsupervised training (so getbatch() will not return uids).
    // This mode requires utterances with time stamps.
    Bundler::Bundler(
        const ConfigParameters &,
        const std::vector<std::vector<wstring>> & infiles,
        const std::vector<map<wstring, std::vector<msra::asr::htkmlfentry>>> & labels,
        std::vector<size_t> vdim,
        std::vector<size_t> udim,
        std::vector<size_t> leftcontext,
        std::vector<size_t> rightcontext,
        size_t /*randomizationrange*/,
        const latticesource & lattices,
        const map<wstring, msra::lattices::lattice::htkmlfwordsequence> & allwordtranscripts,
        const bool framemode,
        std::vector<FrameDescription> featureFrameDescriptions,
        std::vector<FrameDescription> labelFrameDescriptions,
        std::vector<InputDescriptionPtr> inputs,
        std::map<std::wstring, size_t> nameToId,
        std::map<std::wstring, size_t> featureNameToIdMap,
        std::map<std::wstring, size_t> labelNameToIdMap,
        size_t elementSize)
        : m_vdim(vdim)
        , m_udim(udim)
        , m_leftcontext(leftcontext)
        , m_rightcontext(rightcontext)
        , m_sampperiod(0)
        , m_featdim(0)
        , m_lattices(lattices)
        , m_allwordtranscripts(allwordtranscripts)
        , m_framemode(framemode)
        , m_chunksinram(0)
        , m_timegetbatch(0)
        , m_verbosity(2)
        , m_featureFrameDescriptions(featureFrameDescriptions)
        , m_labelFrameDescriptions(labelFrameDescriptions)
        , m_inputs(inputs)
        , m_nameToId(nameToId)
        , m_featureNameToIdMap(featureNameToIdMap)
        , m_labelNameToIdMap(labelNameToIdMap)
        , m_elementSize(elementSize)
        // [v-hansu] change framemode (lattices.empty()) into framemode (false) to run utterance mode without lattice
        // you also need to change another line, search : [v-hansu] comment out to run utterance mode without lattice
    {
        // process infiles to know dimensions of things (but not loading features)
        size_t nomlf = 0;                       // number of entries missing in MLF (diagnostics)
        size_t nolat = 0;                       // number of entries missing in lattice archive (diagnostics)
        std::vector<size_t> numclasses;                  // number of output classes as found in the label file (diagnostics)
        m_totalframes = 0;
        wstring key;
        size_t numutts = 0;

        std::vector<bool> uttisvalid; // boolean flag to check that utterance is valid. valid means number of
        //frames is consistent across all feature and label streams
        std::vector<size_t> uttduration; // track utterance durations to determine utterance validity

        std::vector<size_t> classidsbegin;

        assert(infiles.size() == 1); // we are only looking at a single file here...
        assert(leftcontext.size() == 1);
        assert(leftcontext[0] == 0);
        assert(rightcontext.size() == 1);
        assert(rightcontext[0] == 0);
        assert(labels.size() == 1); // only have one

        m_allchunks = std::vector<std::vector<utterancechunkdata>>(infiles.size(), std::vector<utterancechunkdata>());
        m_featdim = std::vector<size_t>(infiles.size(), 0);
        m_sampperiod = std::vector<unsigned int>(infiles.size(), 0);
        m_featkind = std::vector<string>(infiles.size(), "");

        numclasses = std::vector<size_t>(labels.size(), 0);
        m_counts = std::vector<std::vector<size_t>>(labels.size(), std::vector<size_t>());

        foreach_index(i, labels)
        {
            m_classids.push_back(unique_ptr<biggrowablevector<CLASSIDTYPE>>(new biggrowablevector<CLASSIDTYPE>()));
            m_phoneboundaries.push_back(unique_ptr<biggrowablevector<HMMIDTYPE>>(new biggrowablevector<HMMIDTYPE>()));
            //std::pair<std::vector<wstring>,std::vector<wstring>> latticetocs;
            //std::unordered_map<std::string,size_t> modelsymmap;
            //lattices.push_back(shared_ptr<latticesource>(new latticesource(latticetocs, modelsymmap)));
        }


        // first check consistency across feature streams
        // We'll go through the SCP files for each stream to make sure the duration is consistent
        // If not, we'll plan to ignore the utterance, and inform the user
        // m indexes the feature stream
        // i indexes the files within a stream, i.e. in the SCP file
        numutts = infiles[0].size();
        uttisvalid = std::vector<bool>(numutts, true);
        uttduration = std::vector<size_t>(numutts, 0);

        foreach_index(m, infiles) 
        {
            if (infiles[m].size() != numutts)
            {
                RuntimeError("minibatchutterancesourcemulti: all feature files must have same number of utterances");
            }

            foreach_index(i, infiles[m])
            {
                utterancedesc utterance(msra::asr::htkfeatreader::parsedpath(infiles[m][i]), 0);  //mseltzer - is this foolproof for multiio? is classids always non-empty?
                const size_t uttframes = utterance.numframes(); // will throw if frame bounds not given --required to be given in this mode

                // we need at least 2 frames for boundary markers to work
                if (uttframes < 2 || uttframes > 65535 /* TODO frameref::maxframesperutterance */)
                {
                    fprintf(stderr, "minibatchutterancesource: skipping %d-th file (%d frames) because it exceeds max. frames (%d) for frameref bit field: %ls\n", i, (int)uttframes, (int)65535 /* frameref::maxframesperutterance */, key.c_str());
                    uttduration[i] = 0;
                    uttisvalid[i] = false;
                }
                else
                {
                    if (m == 0)
                    {
                        uttduration[i] = uttframes;
                        uttisvalid[i] = true;
                    }
                    else if (uttduration[i] != uttframes)
                    {
                        fprintf(stderr, "minibatchutterancesource: skipping %d-th file due to inconsistency in duration in different feature streams (%d vs %d frames)\n", i, (int)uttduration[i], (int)uttframes);
                        uttduration[i] = 0;
                        uttisvalid[i] = false;
                    }
                }
            }
        }

        // shouldn't this be checked (again) later? more utterances can be invalidated...
        size_t invalidutts = 0;
        foreach_index(i, uttisvalid) {
            if (!uttisvalid[i])
                invalidutts++;
        }
        assert(invalidutts == 0); // For us it's zero
        if (invalidutts > uttisvalid.size() / 2)
            RuntimeError("minibatchutterancesource: too many files with inconsistent durations, assuming broken configuration\n");
        else if (invalidutts > 0)
            fprintf(stderr, "Found inconsistent durations across feature streams in %d out of %d files\n", (int)invalidutts, (int)uttisvalid.size());


        // now process the features and labels
        size_t utterancesetsize = 0;
        foreach_index(m, infiles)
        {
            std::vector<utterancedesc> utteranceset;// read all utterances to here first; at the end, distribute to chunks
            utteranceset.reserve(infiles[m].size());
            //if (m==0)
            //    numutts = infiles[m].size();
            //else
            //    if (infiles[m].size()!=numutts)
            //        RuntimeError("minibatchutterancesourcemulti: all feature files must have same number of utterances\n");
            if (m == 0)
                classidsbegin.clear();

            foreach_index(i, infiles[m])
            {
                if (i % (infiles[m].size() / 100 + 1) == 0) { fprintf(stderr, "."); fflush(stderr); }
                // build utterance descriptor
                if (m == 0 && !labels.empty())
                    classidsbegin.push_back(m_classids[0]->size());

                if (uttisvalid[i])
                {
                    utterancedesc utterance(msra::asr::htkfeatreader::parsedpath(infiles[m][i]), labels.empty() ? 0 : classidsbegin[i]);  //mseltzer - is this foolproof for multiio? is classids always non-empty?
                    const size_t uttframes = utterance.numframes(); // will throw if frame bounds not given --required to be given in this mode
                    assert(uttframes == uttduration[i]); // ensure nothing funky happened
                    // already performed these checks above
                    // we need at least 2 frames for boundary markers to work
                    //if (uttframes < 2)
                    //    RuntimeError("minibatchutterancesource: utterances < 2 frames not supported");
                    //if (uttframes > frameref::maxframesperutterance)
                    //{
                    //    fprintf (stderr, "minibatchutterancesource: skipping %d-th file (%d frames) because it exceeds max. frames (%d) for frameref bit field: %ls", i, uttframes, frameref::maxframesperutterance, key.c_str());
                    //    continue;
                    //}

                    // check whether we have the ref transcript
                    bool lacksmlf = true;
                    if (!labels.empty())    // empty means unsupervised mode (don't load any)
                    {
                        key = utterance.key();
                        // check if labels are available (if not, it normally means that no path was found in realignment)
                        auto labelsiter = labels[0].find(key);
                        //const bool lacksmlf = (labelsiter == labels[0].end());
                        lacksmlf = (labelsiter == labels[0].end());
                        if (lacksmlf)
                            if (nomlf++ < 5)
                                fprintf(stderr, " [no labels for  %ls]", key.c_str());
                        // check if lattice is available (when in lattice mode)
                        // TODO: also check the #frames here; requires a design change of the TOC format & a rerun
                        const bool lackslat = !lattices.empty() && !lattices.haslattice(key); // ('true' if we have no lattices)
                        if (lackslat)
                            if (nolat++ < 5)
                                fprintf(stderr, " [no lattice for %ls]", key.c_str());
                        // skip if either one is missing
                        if (lacksmlf || lackslat){
                            uttisvalid[i] = false;
                            continue;   // skip this utterance at all
                        }
                    }
                    // push the label sequence into classids[], since we already looked it up
                    // TODO: we can store labels more efficiently now since we don't do frame-wise random access anymore.

                    // OK, utterance has all we need --remember it

                    if (m == 0)
                    {
                        if (!labels.empty() && !lacksmlf)
                        {
                            // first verify that all the label files have the proper duration
                            foreach_index(j, labels)
                            {
                                const auto & labseq = labels[j].find(key)->second;
                                // check if durations match; skip if not
                                size_t labframes = labseq.empty() ? 0 : (labseq[labseq.size() - 1].firstframe + labseq[labseq.size() - 1].numframes);
                                if (labframes != uttframes)
                                {
                                    fprintf(stderr, " [duration mismatch (%d in label vs. %d in feat file), skipping %ls]", (int)labframes, (int)uttframes, key.c_str());
                                    nomlf++;
                                    uttisvalid[i] = false;
                                    //continue;   // skip this utterance at all
                                    break;
                                }
                            }
                            if (uttisvalid[i])
                            {
                                utteranceset.push_back(std::move(utterance));
                                m_totalframes += uttframes;
                                // then parse each mlf if the durations are consistent
                                foreach_index(j, labels)
                                {
                                    const auto & labseq = labels[j].find(key)->second;
                                    // expand classid sequence into flat array
                                    foreach_index(i, labseq)
                                    {
                                        const auto & e = labseq[i];
                                        if ((i > 0 && labseq[i - 1].firstframe + labseq[i - 1].numframes != e.firstframe) || (i == 0 && e.firstframe != 0))
                                        {
                                            RuntimeError("minibatchutterancesource: labels not in consecutive order MLF in label set: %ls", key.c_str());
                                            // TODO Why will these yield a run-time error as opposed to making the utterance invalid?
                                            // TODO check this at the source. Saves storing numframes field.
                                        }
                                        if (e.classid >= udim[j])
                                        {
                                            RuntimeError("minibatchutterancesource: class id %d exceeds model output dimension %d in file %ls", (int)e.classid, (int)udim[j], key.c_str());
                                        }
                                        if (e.classid != (CLASSIDTYPE)e.classid)
                                            RuntimeError("CLASSIDTYPE has too few bits");
                                        for (size_t t = e.firstframe; t < e.firstframe + e.numframes; t++)
                                        {
                                            m_classids[j]->push_back(e.classid);
                                            if (e.phonestart != 0 && t == e.firstframe)
                                                m_phoneboundaries[j]->push_back((HMMIDTYPE)e.phonestart);
                                            else
                                                m_phoneboundaries[j]->push_back((HMMIDTYPE)0);
                                        }
                                        numclasses[j] = max(numclasses[j], (size_t)(1u + e.classid));
                                        m_counts[j].resize(numclasses[j], 0);
                                        m_counts[j][e.classid] += e.numframes;
                                    }

                                    m_classids[j]->push_back((CLASSIDTYPE)-1);  // append a boundary marker marker for checking
                                    m_phoneboundaries[j]->push_back((HMMIDTYPE)-1); // append a boundary marker marker for checking

                                    if (!labels[j].empty() && m_classids[j]->size() != m_totalframes + utteranceset.size())
                                        LogicError("minibatchutterancesource: label duration inconsistent with feature file in MLF label set: %ls", key.c_str());
                                    assert(labels[j].empty() || m_classids[j]->size() == m_totalframes + utteranceset.size());
                                }
                            }
                        }
                        else
                        {
                            assert(m_classids.empty() && labels.empty());
                            utteranceset.push_back(std::move(utterance));
                            m_totalframes += uttframes;
                        }
                    }
                    else
                    {
                        utteranceset.push_back(std::move(utterance));
                    }
                }
            }
            if (m == 0)
                utterancesetsize = utteranceset.size();
            else
                assert(utteranceset.size() == utterancesetsize);

            fprintf(stderr, "feature set %d: %d frames in %d out of %d utterances\n", m, (int)m_totalframes, (int)utteranceset.size(), (int)infiles[m].size());

            if (!labels.empty()){
                foreach_index(j, labels){
                    biggrowablevector<CLASSIDTYPE> & cid = *m_classids[j];
                    foreach_index(i, utteranceset){
                        //if ((*classids[j])[utteranceset[i].classidsbegin + utteranceset[i].numframes()] != (CLASSIDTYPE) -1)
                        //printf("index = %d\n",utteranceset[i].classidsbegin + utteranceset[i].numframes());
                        //printf("cid[index] = %d\n",cid[utteranceset[i].classidsbegin + utteranceset[i].numframes()]);
                        //printf("CLASSIDTYPE(-1) = %d\n",(CLASSIDTYPE) -1);
                        if (cid[utteranceset[i].classidsbegin + utteranceset[i].numframes()] != (CLASSIDTYPE)-1)
                            LogicError("minibatchutterancesource: classids[] out of sync");
                    }
                }
            }
            if (nomlf + nolat > 0)
            {
                fprintf(stderr, "minibatchutterancesource: out of %d files, %d files not found in label set and %d have no lattice\n", (int)infiles[0].size(), (int)nomlf, (int)nolat);
                if (nomlf + nolat > infiles[m].size() / 2)
                    RuntimeError("minibatchutterancesource: too many files not found in label set--assuming broken configuration\n");
            }
            assert(nomlf + nolat == 0); // For us it's zero
            if (m == 0) { foreach_index(j, numclasses) { fprintf(stderr, "label set %d: %d classes\n", j, (int)numclasses[j]); } }
            // distribute them over chunks
            // We simply count off frames until we reach the chunk size.
            // Note that we first randomize the chunks, i.e. when used, chunks are non-consecutive and thus cause the disk head to seek for each chunk.
            const size_t framespersec = 100;                    // we just assume this; our efficiency calculation is based on this
            const size_t chunkframes = 15 * 60 * framespersec;  // number of frames to target for each chunk
            // Loading an initial 24-hour range will involve 96 disk seeks, acceptable.
            // When paging chunk by chunk, chunk size ~14 MB.
            std::vector<utterancechunkdata> & thisallchunks = m_allchunks[m];

            thisallchunks.resize(0);
            thisallchunks.reserve(m_totalframes / chunkframes); // This is ignoring I/O for invalid utterances... // TODO round up?

            foreach_index(i, utteranceset)
            {
                // if exceeding current entry--create a new one
                // I.e. our chunks are a little larger than wanted (on av. half the av. utterance length).
                if (thisallchunks.empty() || thisallchunks.back().totalframes > chunkframes || thisallchunks.back().numutterances() >= 65535 /* TODO frameref::maxutterancesperchunk */)
                    // TODO > instead of >= ? if (thisallchunks.empty() || thisallchunks.back().totalframes > chunkframes || thisallchunks.back().numutterances() >= frameref::maxutterancesperchunk)
                    thisallchunks.push_back(utterancechunkdata());
                // append utterance to last chunk
                utterancechunkdata & currentchunk = thisallchunks.back();
                currentchunk.push_back(std::move(utteranceset[i]));    // move it out from our temp array into the chunk
                // TODO: above push_back does not actually 'move' because the internal push_back does not accept that
            }
            m_numutterances = utteranceset.size();
            fprintf(stderr, "minibatchutterancesource: %d utterances grouped into %d chunks, av. chunk size: %.1f utterances, %.1f frames\n",
                (int)m_numutterances, (int)thisallchunks.size(), m_numutterances / (double)thisallchunks.size(), m_totalframes / (double)thisallchunks.size());
            // Now utterances are stored exclusively in allchunks[]. They are never referred to by a sequential utterance id at this point, only by chunk/within-chunk index.

            // Initialize the block randomizer
            //rand = std::make_unique<BlockRandomizer>(m_verbosity, framemode, m_totalframes, m_numutterances, randomizationrange, nullptr);
        }

        size_t sequenceId = 0;
        const std::vector<utterancechunkdata>& chunks = m_allchunks[0];
        foreach_index(i, chunks)
        {
            foreach_index(j, chunks[i].utteranceset)
            {
                if (framemode)
                {
                    for (size_t k = 0; k < chunks[i].utteranceset[j].numframes(); ++k)
                    {
                        SequenceDescription description;
                        description.id = sequenceId++;
                        description.chunkId = i;
                        description.numberOfSamples = 1;
                        m_sequenceIdToSequence.insert(std::make_pair(description.id, &chunks[i].utteranceset[j]));
                        m_timeline.push_back(description);

                        auto sq = BlockRandomizer::sequenceref(i, j, k);
                        sq.numframes = 1;
                        m_sequences.push_back(sq);
                    }
                }
                else
                {
                    SequenceDescription description;
                    description.id = sequenceId++;
                    description.chunkId = i;
                    description.numberOfSamples = chunks[i].utteranceset[j].numframes();
                    m_sequenceIdToSequence.insert(std::make_pair(description.id, &chunks[i].utteranceset[j]));
                    m_timeline.push_back(description);

                    auto sq = BlockRandomizer::sequenceref(i, j, 0);
                    sq.numframes = description.numberOfSamples;
                    m_sequences.push_back(sq);
                }
            }
        }
    }

    bool Bundler::RequireChunk(size_t chunkindex)
    {
        size_t numinram = 0;

        const size_t numStreams = m_allchunks.size();
        for (size_t m = 0; m < numStreams; m++)
        {
            auto & chunkdata = m_allchunks[m][chunkindex];
            if (chunkdata.isinram())
                numinram++;
        }
        if (numinram == numStreams)
            return false;
        else if (numinram == 0)
        {
            for (size_t m = 0; m < numStreams; m++)
            {
                auto & chunkdata = m_allchunks[m][chunkindex];
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

    void Bundler::ReleaseChunk(size_t chunkIndex)
    {
        size_t numreleased = 0;
        const size_t numStreams = m_allchunks.size();
        for (size_t m = 0; m < numStreams; m++)
        {
            auto & chunkdata = m_allchunks[m][chunkIndex];
            if (chunkdata.isinram())
            {
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
    }

    const Timeline& Bundler::getTimeline() const
    {
        return m_timeline;
    }

    std::vector<InputDescriptionPtr> Bundler::getInputs() const
    {
        return m_inputs;
    }

    SequenceData Bundler::getSequenceById(size_t id)
    {
        SequenceData result;

        std::vector<msra::dbn::matrix> feat;              // buffer for holding curernt minibatch's frames
        std::vector<std::vector<size_t>> uids;               // buffer for storing current minibatch's frame-level label sequence

        auto_timer timergetbatch;
        assert(m_totalframes > 0);

        const size_t numChunks = m_allchunks[0].size();
        const size_t numStreams = m_allchunks.size();

        const std::vector<char> noboundaryflags;    // dummy

        const size_t spos = id; // positer->second;
        const size_t epos = spos + 1;


        // Determine window range
        const size_t windowbegin = m_rand->getSequenceWindowBegin(id);
        const size_t windowend = m_rand->getSequenceWindowEnd(id);

        for (size_t k = 0; k < windowbegin; k++)
        {
            ReleaseChunk(k);
        }

        for (size_t k = windowend; k < numChunks; k++)
        {
            ReleaseChunk(k);
        }

        for (size_t pos = spos; pos < epos; pos++)
        {
            if (m_timeline[id].chunkId % m_numberOfWorkers == m_workerRank)
            {
                RequireChunk(m_timeline[id].chunkId); // (window range passed in for checking only)
            }
        }

        // Note that the above loop loops over all chunks incl. those that we already should have.
        // This has an effect, e.g., if 'numsubsets' has changed (we will fill gaps).

        // determine the true #frames we return, for allocation--it is less than mbframes in the case of MPI/data-parallel sub-set mode
        size_t tspos = 0;
        for (size_t pos = spos; pos < epos; pos++)
        {
            auto chunkId = m_timeline[id].chunkId;
            if ((chunkId % m_numberOfWorkers) != m_workerRank)            // chunk not to be returned for this MPI node
            {
                continue;
            }

            tspos += m_timeline[id].numberOfSamples;
        }

        if (tspos == 0)
        {
            return result;
        }

        // resize feat and uids
        // eldak:s should return phone boundaries and sentendmark lattices transcripts etc.
        feat.resize(m_vdim.size());
        uids.resize(m_classids.size());
        //phoneboundaries.resize(m_classids.size());
        //sentendmark.resize(m_vdim.size());
        //assert(feat.size() == vdim.size());
        //assert(feat.size() == randomizedchunks.size());
        foreach_index(i, feat)
        {
            feat[i].resize(m_vdim[i], tspos);

            if (i == 0)
            {
                foreach_index(j, uids)
                {
                    if (issupervised())             // empty means unsupervised training -> return empty uids
                    {
                        uids[j].resize(tspos);
                        //phoneboundaries[j].resize(tspos);
                    }
                    else
                    {
                        uids[i].clear();
                        //phoneboundaries[i].clear();
                    }
                    //latticepairs.clear();               // will push_back() below
                    //transcripts.clear();
                }
                //foreach_index(j, sentendmark)
                //{
                //    sentendmark[j].clear();
                //}
            }
        }
        //// return these utterances
        //if (verbosity > 0)
        //    fprintf(stderr, "getbatch: getting utterances %d..%d (%d subset of %d frames out of %d requested) in sweep %d\n", (int)spos, (int)(epos - 1), (int)tspos, (int)mbframes, (int)framesrequested, (int)sweep);
        tspos = 0;   // relative start of utterance 'pos' within the returned minibatch
        size_t numberOfFrames = 0; // eldak: seems this should be changed for sequences though.
        for (size_t pos = spos; pos < epos; pos++)
        {
            const auto& sequence = m_timeline[id];
            const auto & uttref = m_sequences[id];

            if ((uttref.chunkindex % m_numberOfWorkers) != m_workerRank)            // chunk not to be returned for this MPI node
            {
                continue;
            }

            size_t n = 0;
            for (size_t i = 0; i < numStreams; ++i)
            {
                const auto & chunkdata = m_allchunks[i][uttref.chunkindex];

                // eldak - does it mean we have read the who?
                // assert((m_numberOfWorkers > 1) || (uttref.globalts == globalts + tspos));
                auto uttframes = chunkdata.getutteranceframes(uttref.utteranceindex);
                matrixasvectorofvectors uttframevectors(uttframes);    // (wrapper that allows m[j].size() and m[j][i] as required by augmentneighbors())
                n = uttframevectors.size();

                //sentendmark[i].push_back(n + tspos);
                assert(n == uttframes.cols() &&
                      (uttref.numframes == n && !m_framemode || (uttref.numframes == 1 && m_framemode)) &&
                      (chunkdata.numframes(uttref.utteranceindex) == n && !m_framemode || (uttref.numframes == 1 && m_framemode)));

                // copy the frames and class labels
                for (size_t t = 0; t < sequence.numberOfSamples; t++)          // t = time index into source utterance
                {
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

                    augmentneighbors(uttframevectors, noboundaryflags, uttref.frameindex + t, leftextent, rightextent, feat[i], t + tspos);
                }

                // copy the frames and class labels
                if (i == 0)
                {
                    auto uttclassids = GetClassIds(uttref);
                    //auto uttphoneboudaries = getphonebound(uttref);
                    foreach_index(j, uttclassids)
                    {
                        for (size_t t = 0; t < sequence.numberOfSamples; t++)          // t = time index into source utterance
                        {
                            if (issupervised())
                            {
                                uids[j][t + tspos] = uttclassids[j][uttref.frameindex + t];
                                //phoneboundaries[j][t + tspos] = uttphoneboudaries[j][t];
                            }
                        }

                        // eldak - no lattices currently.
                        //if (!this->lattices.empty())
                        //{
                        //    auto latticepair = chunkdata.getutterancelattice(uttref.utteranceindex);
                        //    latticepairs.push_back(latticepair);
                        //    // look up reference
                        //    const auto & key = latticepair->getkey();
                        //    if (!allwordtranscripts.empty())
                        //    {
                        //        const auto & transcript = allwordtranscripts.find(key)->second;
                        //        transcripts.push_back(transcript.words);
                        //    }
                        //}
                    }
                }
            }
            tspos += sequence.numberOfSamples;
            numberOfFrames++;
        }

        foreach_index(i, feat)
        {
            assert(tspos == feat[i].cols());
        }

        m_timegetbatch = timergetbatch;

        for (auto it = m_featureNameToIdMap.begin(); it != m_featureNameToIdMap.end(); ++it)
        {
            Sequence r;
            size_t id = m_featureNameToIdMap[it->first];

            // eldak: leak here.
            const msra::dbn::matrixstripe featOri = msra::dbn::matrixstripe(feat[id], 0, feat[0].cols());
            const size_t dimensions = featOri.rows();
            const void* tmp = &featOri(0, 0);

            r.numberOfFrames = 1;
            r.frameDescription = &m_featureFrameDescriptions[id];

            // eldak: leak leak leak. who is responsible for clearing this? who does caching?
            void* buffer = nullptr;
            if (m_elementSize == sizeof(float))
            {
                buffer = new float[featOri.rows()];
            }
            else
            {
                buffer = new double[featOri.rows()];
            }

            memset(buffer, 0, m_elementSize * dimensions);
            memcpy_s(buffer, m_elementSize * dimensions, tmp, m_elementSize * dimensions);
            r.data = buffer;

            result.m_data.insert(std::make_pair(m_nameToId[it->first], r));
        }

        for (auto it = m_labelNameToIdMap.begin(); it != m_labelNameToIdMap.end(); ++it)
        {
            Sequence r;
            size_t id = m_labelNameToIdMap[it->first];
            size_t dim = m_udim[id];

            const std::vector<size_t>& x = uids[id];

            // eldak: leak here.
            if (m_elementSize == sizeof(float))
            {
                float* tmp = new float[dim];
                memset(tmp, 0, m_elementSize * dim);
                tmp[x[0]] = 1;
                r.data = tmp;
                r.numberOfFrames = 1;
                r.frameDescription = &m_labelFrameDescriptions[id];
            }
            else
            {
                double* tmp = new double[dim];
                tmp[x[0]] = 1;
                r.data = tmp;
                r.numberOfFrames = 1;
                r.frameDescription = &m_labelFrameDescriptions[id];
            }
            result.m_data.insert(std::make_pair(m_nameToId[it->first], r));
        }

        return result;
    }

    void Bundler::SetEpochConfiguration(const Microsoft::MSR::CNTK::EpochConfiguration& config)
    {
        m_workerRank = config.workerRank;
        m_numberOfWorkers = config.numberOfWorkers;
    }
}}
