#include "stdafx.h"
#include "Bundler.h"

namespace msra { namespace dbn {

    // constructor
    // Pass empty labels to denote unsupervised training (so getbatch() will not return uids).
    // This mode requires utterances with time stamps.
    Bundler::Bundler(
        const std::vector<std::vector<wstring>> & infiles,
        const std::vector<map<wstring, std::vector<msra::asr::htkmlfentry>>> & labels,
        std::vector<size_t> vdim,
        std::vector<size_t> udim,
        std::vector<size_t> leftcontext,
        std::vector<size_t> rightcontext,
        size_t randomizationrange,
        const latticesource & lattices,
        const map<wstring, msra::lattices::lattice::htkmlfwordsequence> & allwordtranscripts,
        const bool framemode)
        : m_vdim(vdim)
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
        foreach_index(m, infiles) {
            if (m == 0) {
                numutts = infiles[m].size();
                uttisvalid = std::vector<bool>(numutts, true);
                uttduration = std::vector<size_t>(numutts, 0);
            }
            else if (infiles[m].size() != numutts)
                RuntimeError("minibatchutterancesourcemulti: all feature files must have same number of utterances");

            foreach_index(i, infiles[m]){
                utterancedesc utterance(msra::asr::htkfeatreader::parsedpath(infiles[m][i]), 0);  //mseltzer - is this foolproof for multiio? is classids always non-empty?
                const size_t uttframes = utterance.numframes(); // will throw if frame bounds not given --required to be given in this mode
                // we need at least 2 frames for boundary markers to work
                if (uttframes < 2)
                    RuntimeError("minibatchutterancesource: utterances < 2 frames not supported");
                if (uttframes > 65535 /* TODO frameref::maxframesperutterance */)
                {
                    fprintf(stderr, "minibatchutterancesource: skipping %d-th file (%d frames) because it exceeds max. frames (%d) for frameref bit field: %ls\n", i, (int)uttframes, (int)65535 /* frameref::maxframesperutterance */, key.c_str());
                    uttduration[i] = 0;
                    uttisvalid[i] = false;
                }
                else
                {
                    if (m == 0){
                        uttduration[i] = uttframes;
                        uttisvalid[i] = true;
                    }
                    else if (uttduration[i] != uttframes){
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
            rand = std::make_unique<BlockRandomizer>(m_verbosity, framemode, m_totalframes, m_numutterances, randomizationrange);
        }
    }
}}