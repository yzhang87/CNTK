#include "stdafx.h"
#include "BundlerSplitted.h"
#include <DataReader.h>
#include "Utils.h"
#include "ConfigHelper.h"
#include "msra_mgram.h"
#include <DataTensor.h>
#include "HTKDataDeserializer.h"
#include "MLFDataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // constructor
    // Pass empty labels to denote unsupervised training (so getbatch() will not return uids).
    void BundlerSplitted::OldInit(
        const ConfigParameters& readerConfig,
        bool framemode,
        size_t elementSize,
        int verbosity)
    {
        m_framemode = framemode;
        m_chunksinram = 0;
        m_verbosity = verbosity;
        m_elementSize = elementSize;

        std::vector<std::wstring> featureNames;
        std::vector<std::wstring> labelNames;

        std::vector<std::wstring> notused;
        Utils::GetDataNamesFromConfig(readerConfig, featureNames, labelNames, notused, notused);
        if (featureNames.size() < 1 || labelNames.size() < 1)
        {
            InvalidArgument("network needs at least 1 input and 1 output specified!");
        }

        std::vector<InputDescriptionPtr> inputs;
        std::vector<size_t> contextLeft;
        std::vector<size_t> contextRight;
        std::vector<std::vector<std::wstring>> featurePaths; /// TODO get rid
        foreach_index(i, featureNames) // TODO only need range-based
        {
            InputDescriptionPtr input = std::make_shared<InputDescription>();
            input->id = inputs.size();
            inputs.push_back(input);

            const std::wstring& featureName = featureNames[i];
            input->name = featureName;

            const ConfigParameters& thisFeature = readerConfig(featureName);
            ConfigHelper::CheckFeatureType(thisFeature);

            auto context = ConfigHelper::GetContextWindow(thisFeature);
            contextLeft.push_back(context.first);
            contextRight.push_back(context.second);

            size_t dim = thisFeature(L"dim");
            dim = dim * (1 + context.first + context.second);

            SampleLayoutPtr layout = std::make_shared<ImageLayout>(std::move(std::vector<size_t>{ dim }));
            input->sampleLayout = layout;

            // eldak TODO: check that all script files for multi io have the same length.
            featurePaths.push_back(std::move(ConfigHelper::GetFeaturePaths(thisFeature)));
            m_featureIndices.push_back(input->id);

            auto deserializer = std::make_shared<HTKDataDeserializer>(thisFeature, m_elementSize);
            m_featureDeserializers.push_back(deserializer);
        }

        std::vector<std::wstring> stateListPaths;
        std::vector<std::vector<std::wstring>> mlfPaths;
        foreach_index(i, labelNames)
        {
            InputDescriptionPtr input = std::make_shared<InputDescription>();
            input->id = inputs.size();
            inputs.push_back(input);

            const std::wstring& labelName = labelNames[i];
            input->name = labelName;

            const ConfigParameters& thisLabel = readerConfig(labelName);
            ConfigHelper::CheckLabelType(thisLabel);

            size_t dim = ConfigHelper::GetLabelDimension(thisLabel);

            SampleLayoutPtr layout = std::make_shared<ImageLayout>(std::move(std::vector<size_t> { dim }));
            input->sampleLayout = layout;

            stateListPaths.push_back(thisLabel(L"labelMappingFile", L""));

            mlfPaths.push_back(std::move(ConfigHelper::GetMlfPaths(thisLabel)));
            m_labelIndices.push_back(input->id);

            auto deserializer = std::make_shared<MLFDataDeserializer>(thisLabel, m_elementSize);
            m_labelDeserializers.push_back(deserializer);
        }

        assert(featurePaths.size() == m_featureIndices.size());
        assert(mlfPaths.size() == m_labelIndices.size());

        m_verbosity = readerConfig(L"verbosity", 2);

        m_sequenceIdTolabelId = std::vector<std::vector<size_t>>(m_labelDeserializers.size(), std::vector<size_t>());

        //look up in the config for the master command to determine whether we're writing output (inputs only) or training/evaluating (inputs and outputs)
        wstring command(readerConfig(L"action", L""));

        //eldak TODO: does not belong here
        std::wstring readMethod = ConfigHelper::GetRandomizer(readerConfig);
        if (_wcsicmp(readMethod.c_str(), L"blockRandomize"))
        {
            RuntimeError("readMethod must be 'blockRandomize'");
        }

        m_verbosity = readerConfig(L"verbosity", 2);

        // currently assumes all Mlfs will have same root name (key)
        set<wstring> restrictmlftokeys;     // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files

        // get labels
        double htktimetoframe = 100000.0; // default is 10ms
        std::vector<std::map<std::wstring, std::vector<msra::asr::htkmlfentry>>> labelsmulti;

        foreach_index(i, mlfPaths)
        {
            msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>
                l(mlfPaths[i], restrictmlftokeys, stateListPaths[i], (const msra::lm::CSymbolSet*) NULL, (map<string, size_t>*) NULL, htktimetoframe);
            labelsmulti.push_back(std::move(l));

            // label MLF
            // get the temp file name for the page file

            // Make sure 'msra::asr::htkmlfreader' type has a move constructor
            static_assert(std::is_move_constructible<msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>>::value,
                "Type 'msra::asr::htkmlfreader' should be move constructible!");
        }

        const std::vector<std::vector<wstring>>& infiles = featurePaths;
        const std::vector<map<wstring, std::vector<msra::asr::htkmlfentry>>> & labels = labelsmulti;
        m_leftcontext = contextLeft;
        m_rightcontext = contextRight;
        m_inputs = inputs;
        m_elementSize = elementSize;
        m_featkind = std::vector<string>(infiles.size(), "");
        m_sampperiod = std::vector<unsigned int>(infiles.size(), 0);

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
        assert(m_leftcontext.size() == 1);
        assert(m_leftcontext[0] == 0);
        assert(m_rightcontext.size() == 1);
        assert(m_rightcontext[0] == 0);
        assert(labels.size() == 1); // only have one

        m_allchunks = std::vector<std::vector<utterancechunkdata>>(infiles.size(), std::vector<utterancechunkdata>());
        m_featdim = std::vector<size_t>(infiles.size(), 0);

        numclasses = std::vector<size_t>(labels.size(), 0); // numLabelStreams

        foreach_index(i, labels)
        {
            m_classids.push_back(unique_ptr<msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE>>(new msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE>()));
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
                        const bool lackslat = false;// !lattices.empty() && !lattices.haslattice(key); // ('true' if we have no lattices)
                        /*                        if (lackslat)
                        if (nolat++ < 5)
                        fprintf(stderr, " [no lattice for %ls]", key.c_str());*/
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

                                        auto dimension = m_inputs[m_labelIndices[j]]->sampleLayout->GetDims()[0];

                                        if (e.classid >= dimension) //udim[j])
                                        {
                                            RuntimeError("minibatchutterancesource: class id %d exceeds model output dimension %d in file %ls", (int)e.classid, (int)dimension, key.c_str());
                                        }
                                        if (e.classid != (msra::dbn::CLASSIDTYPE)e.classid)
                                            RuntimeError("CLASSIDTYPE has too few bits");
                                        for (size_t t = e.firstframe; t < e.firstframe + e.numframes; t++)
                                        {
                                            m_classids[j]->push_back(e.classid);
                                        }
                                        numclasses[j] = max(numclasses[j], (size_t)(1u + e.classid));
                                    }

                                    m_classids[j]->push_back((msra::dbn::CLASSIDTYPE) - 1);  // append a boundary marker marker for checking

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
                    msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE> & cid = *m_classids[j];
                    foreach_index(i, utteranceset){
                        //if ((*classids[j])[utteranceset[i].classidsbegin + utteranceset[i].numframes()] != (CLASSIDTYPE) -1)
                        //printf("index = %d\n",utteranceset[i].classidsbegin + utteranceset[i].numframes());
                        //printf("cid[index] = %d\n",cid[utteranceset[i].classidsbegin + utteranceset[i].numframes()]);
                        //printf("CLASSIDTYPE(-1) = %d\n",(CLASSIDTYPE) -1);
                        if (cid[utteranceset[i].classidsbegin + utteranceset[i].numframes()] != (msra::dbn::CLASSIDTYPE) - 1)
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

            fprintf(stderr, "minibatchutterancesource: %llu utterances grouped into %llu chunks, av. chunk size: %.1f utterances, %.1f frames\n",
                utteranceset.size(), thisallchunks.size(), utteranceset.size() / (double)thisallchunks.size(), m_totalframes / (double)thisallchunks.size());
            // Now utterances are stored exclusively in allchunks[]. They are never referred to by a sequential utterance id at this point, only by chunk/within-chunk index.
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

                        auto sq = sequenceref(i, j, k);
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

                    auto sq = sequenceref(i, j, 0);
                    sq.numframes = description.numberOfSamples;
                    m_sequences.push_back(sq);
                }
            }
        }
    }

    BundlerSplitted::BundlerSplitted(
        const ConfigParameters& readerConfig,
        bool framemode,
        size_t elementSize,
        int verbosity)
    {
        m_framemode = framemode;
        m_chunksinram = 0;
        m_verbosity = readerConfig(L"verbosity", 2);
        m_verbosity = verbosity; // not needed
        m_elementSize = elementSize;

        std::vector<std::wstring> featureNames;
        std::vector<std::wstring> labelNames;

        std::vector<std::wstring> notused;
        Utils::GetDataNamesFromConfig(readerConfig, featureNames, labelNames, notused, notused);
        if (featureNames.size() < 1 || labelNames.size() < 1)
        {
            // eldak: Don't we support unsupervised training?
            InvalidArgument("network needs at least 1 input and 1 output specified!");
        }

        std::vector<InputDescriptionPtr> inputs;
        for (const auto& featureName : featureNames)
        {
            auto deserializer = std::make_shared<HTKDataDeserializer>(readerConfig(featureName), m_elementSize);
            m_featureDeserializers.push_back(deserializer);

            // eldak: should we simply delegate this to the data deserializer?
            // who sets the id then?
            InputDescriptionPtr input = std::make_shared<InputDescription>();
            input->id = inputs.size();
            input->name = featureName;
            input->sampleLayout = deserializer->GetInput()->sampleLayout;

            inputs.push_back(input);

            m_featureIndices.push_back(input->id);
        }

        for (const auto& labelName : labelNames)
        {
            auto deserializer = std::make_shared<MLFDataDeserializer>(readerConfig(labelName), m_elementSize);
            m_labelDeserializers.push_back(deserializer);

            // eldak: should we simply delegate this to the data deserializer?
            // who sets the id then?
            InputDescriptionPtr input = std::make_shared<InputDescription>();
            input->id = inputs.size();
            input->name = labelName;
            input->sampleLayout = deserializer->GetInput()->sampleLayout;
            inputs.push_back(input);

            m_labelIndices.push_back(input->id);
        }

        //eldak TODO: does not belong here
        std::wstring readMethod = ConfigHelper::GetRandomizer(readerConfig);
        if (_wcsicmp(readMethod.c_str(), L"blockRandomize"))
        {
            RuntimeError("readMethod must be 'blockRandomize'");
        }

        m_sequenceIdTolabelId = std::vector<std::vector<size_t>>(m_labelDeserializers.size(), std::vector<size_t>());

        //const std::vector<std::vector<wstring>>& infiles = featurePaths;
        //const std::vector<map<wstring, std::vector<msra::asr::htkmlfentry>>> & labels = labelsmulti;
        m_inputs = inputs;
        m_elementSize = elementSize;
        //m_featkind = std::vector<string>(infiles.size(), "");
        //m_sampperiod = std::vector<unsigned int>(infiles.size(), 0);

        // process infiles to know dimensions of things (but not loading features)
        size_t nomlf = 0;                       // number of entries missing in MLF (diagnostics)
        //size_t nolat = 0;                       // number of entries missing in lattice archive (diagnostics)
        std::vector<size_t> numclasses;                  // number of output classes as found in the label file (diagnostics)
        m_totalframes = 0;
        wstring key;
        size_t numutts = 0;

        std::vector<size_t> classidsbegin;

        //assert(infiles.size() == 1); // we are only looking at a single file here...

        //m_allchunks = std::vector<std::vector<utterancechunkdata>>(infiles.size(), std::vector<utterancechunkdata>());
        //m_featdim = std::vector<size_t>(infiles.size(), 0);

        numclasses = std::vector<size_t>(m_labelDeserializers.size(), 0);

        /*
        foreach_index(i, labels)
        {
        m_classids.push_back(unique_ptr<msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE>>(new msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE>()));
        }
        */

        const auto& expected = m_featureDeserializers[0]->GetSequenceDescriptions();
        numutts = expected.size();
        std::vector<bool> isValid(numutts, true);

        foreach_index(m, m_featureDeserializers)
        {
            const auto& utterances = m_featureDeserializers[m]->GetSequenceDescriptions();
            if (utterances.size() != numutts)
            {
                RuntimeError("minibatchutterancesourcemulti: all feature files must have same number of utterances");
            }

            foreach_index(i, utterances)
            {
                const SequenceDescription* sequence = utterances[i];
                if (sequence->numberOfSamples != expected[i]->numberOfSamples || !sequence->isValid)
                {
                    //fprintf(stderr, "minibatchutterancesource: skipping %d-th file due to inconsistency in duration in different feature streams (%d vs %d frames)\n", i, (int)uttduration[i], (int)uttframes);
                    isValid[i] = false;
                }
            }
        }

        for (auto& m : m_sequenceIdTolabelId)
        {
            // eldak: possibly not zero but some max size_t?
            m.resize(numutts, 0);
        }

        // shouldn't this be checked (again) later? more utterances can be invalidated...
        size_t invalidUtts = 0;
        foreach_index(i, isValid)
        {
            if (!isValid[i])
            {
                invalidUtts++;
            }
        }
        assert(invalidUtts == 0); // For us it's zero

        if (invalidUtts > isValid.size() / 2)
        {
            RuntimeError("minibatchutterancesource: too many files with inconsistent durations, assuming broken configuration\n");
        }
        else if (invalidUtts > 0)
        {
            fprintf(stderr, "Found inconsistent durations across feature streams in %llu out of %llu files\n", invalidUtts, isValid.size());
        }


        bool isSupervised = !m_labelDeserializers.empty();
        std::vector<std::map<std::wstring, const SequenceDescription*>> labels;
        std::map<std::wstring, const SequenceDescription*>* expectedLabels = nullptr;
        if (isSupervised)
        {
            for (auto d : m_labelDeserializers)
            {
                labels.push_back(std::map<std::wstring, const SequenceDescription*>());
                std::map<std::wstring, const SequenceDescription*>& m = labels[labels.size() - 1];
                for (auto p : d->GetSequenceDescriptions())
                {
                    m[p->key] = p;
                }
            }
            expectedLabels = &labels[0];
        }

        // now process the features and labels
        //size_t utterancesetsize = 0;
        foreach_index(m, m_featureDeserializers)
        {
            const auto& utterances = m_featureDeserializers[m]->GetSequenceDescriptions();

            foreach_index(i, utterances)
            {
                //if (i % (m_featureDeserializers[m].size() / 100 + 1) == 0)
                //{
                //    fprintf(stderr, "."); fflush(stderr);
                //}

                if (!isValid[i])
                {
                    continue;
                }

                //utterancedesc utterance(msra::asr::htkfeatreader::parsedpath(infiles[m][i]), labels.empty() ? 0 : classidsbegin[i]);  //mseltzer - is this foolproof for multiio? is classids always non-empty?
                //const size_t uttframes = utterance.numframes(); // will throw if frame bounds not given --required to be given in this mode

                // check whether we have the ref transcript
                bool lacksmlf = true;
                if (isSupervised)    // empty means unsupervised mode (don't load any)
                {
                    key = utterances[i]->key;

                    auto labelsiter = expectedLabels->find(key);
                    lacksmlf = (labelsiter == expectedLabels->end());

                    if (lacksmlf)
                    {
                        if (nomlf++ < 5)
                        {
                            fprintf(stderr, " [no labels for  %ls]", key.c_str());
                        }

                        isValid[i] = false;
                        continue;   // skip this utterance at all
                    }
                }

                // push the label sequence into classids[], since we already looked it up
                // TODO: we can store labels more efficiently now since we don't do frame-wise random access anymore.

                // OK, utterance has all we need --remember it

                if (m == 0)
                {
                    if (isSupervised)
                    {
                        // first verify that all the label files have the proper duration
                        foreach_index(j, labels)
                        {
                            const auto & labseq = labels[j].find(key)->second;

                            //eldak: assume all labels are aligned, could be not true in general.
                            m_sequenceIdTolabelId[j][utterances[i]->id] = labseq->id;

                            // check if durations match; skip if not
                            //size_t labframes = labseq-> .empty() ? 0 : (labseq[labseq.size() - 1].firstframe + labseq[labseq.size() - 1].numframes);
                            if (labseq->numberOfSamples != utterances[i]->numberOfSamples)
                            {
                                fprintf(
                                    stderr,
                                    " [duration mismatch (%llu in label vs. %llu in feat file), skipping %ls]",
                                    labseq->numberOfSamples,
                                    utterances[i]->numberOfSamples,
                                    key.c_str());

                                nomlf++;
                                isValid[i] = false;
                                //continue;   // skip this utterance at all
                                break;
                            }
                        }

                        if (isValid[i])
                        {
                            m_totalframes += utterances[i]->numberOfSamples;
                            // then parse each mlf if the durations are consistent
                        }
                    }
                    else
                    {
                        assert(m_classids.empty() && labels.empty());
                        m_totalframes += utterances[i]->numberOfSamples;
                    }
                }
            }

            // todo: Should put this into the deserializer itself
            // fprintf(stderr, "feature set %d: %d frames in %d out of %d utterances\n", m, (int)m_totalframes, (int)utteranceset.size(), (int)m_featureDeserializers[m].size());

            if (!labels.empty())
            {
                //foreach_index(j, labels)
                //{
                //    msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE> & cid = *m_classids[j];
                //    foreach_index(i, utterances){
                //        //if ((*classids[j])[utteranceset[i].classidsbegin + utteranceset[i].numframes()] != (CLASSIDTYPE) -1)
                //        //printf("index = %d\n",utteranceset[i].classidsbegin + utteranceset[i].numframes());
                //        //printf("cid[index] = %d\n",cid[utteranceset[i].classidsbegin + utteranceset[i].numframes()]);
                //        //printf("CLASSIDTYPE(-1) = %d\n",(CLASSIDTYPE) -1);
                //        if (cid[utteranceset[i].classidsbegin + utteranceset[i].numframes()] != (msra::dbn::CLASSIDTYPE) - 1)
                //            LogicError("minibatchutterancesource: classids[] out of sync");
                //    }
                //}
            }

            //if (nomlf > 0)
            //{
            //    fprintf(stderr, "minibatchutterancesource: out of %d files, %d files not found in label set and %d have no lattice\n", (int)infiles[0].size(), (int)nomlf, (int)nolat);
            //    if (nomlf + nolat > infiles[m].size() / 2)
            //        RuntimeError("minibatchutterancesource: too many files not found in label set--assuming broken configuration\n");
            //}

            assert(nomlf == 0); // For us it's zero

            //if (m == 0) { foreach_index(j, numclasses) { fprintf(stderr, "label set %d: %d classes\n", j, (int)numclasses[j]); } }

            /*

            // distribute them over chunks
            // We simply count off frames until we reach the chunk size.
            // Note that we first randomize the chunks, i.e. when used, chunks are non-consecutive and thus cause the disk head to seek for each chunk.
            const size_t framespersec = 100;                    // we just assume this; our efficiency calculation is based on this
            const size_t chunkframes = 15 * 60 * framespersec;  // number of frames to target for each chunk
            // Loading an initial 24-hour range will involve 96 disk seeks, acceptable.
            // When paging chunk by chunk, chunk size ~14 MB.


            thisallchunks.resize(0);
            thisallchunks.reserve(m_totalframes / chunkframes); // This is ignoring I/O for invalid utterances... // TODO round up?

            foreach_index(i, utterances)
            {
            // if exceeding current entry--create a new one
            // I.e. our chunks are a little larger than wanted (on av. half the av. utterance length).
            if (thisallchunks.empty() || thisallchunks.back().totalframes > chunkframes || thisallchunks.back().numutterances() >= 65535)
            // TODO > instead of >= ? if (thisallchunks.empty() || thisallchunks.back().totalframes > chunkframes || thisallchunks.back().numutterances() >= frameref::maxutterancesperchunk)
            thisallchunks.push_back(utterancechunkdata());
            // append utterance to last chunk
            utterancechunkdata & currentchunk = thisallchunks.back();
            currentchunk.push_back(utterances[i]);    // move it out from our temp array into the chunk
            // TODO: above push_back does not actually 'move' because the internal push_back does not accept that
            }

            fprintf(stderr, "minibatchutterancesource: %llu utterances grouped into %llu chunks, av. chunk size: %.1f utterances, %.1f frames\n",
            utteranceset.size(), thisallchunks.size(), utteranceset.size() / (double)thisallchunks.size(), m_totalframes / (double)thisallchunks.size());
            // Now utterances are stored exclusively in allchunks[]. They are never referred to by a sequential utterance id at this point, only by chunk/within-chunk index.
            */
        }

        // eldak: currently create the timeline from the feature deserializer.

        TimelineP timeline = m_featureDeserializers[0]->GetSequenceDescriptions();

        if (m_framemode)
        {
            m_timeline.reserve(m_totalframes);
            m_sequenceIdToFeatureId.reserve(m_totalframes);
            m_sequences.reserve(m_totalframes);
        }
        else
        {
            m_timeline.reserve(timeline.size());
            m_sequenceIdToFeatureId.reserve(m_totalframes);
            m_sequences.reserve(m_totalframes);
        }
        


        foreach_index(i, timeline)
        {
            if (m_framemode)
            {
                for (size_t k = 0; k < timeline[i]->numberOfSamples; ++k)
                {
                    SequenceDescription description;
                    description.id = m_timeline.size();
                    m_sequenceIdToFeatureId.push_back(timeline[i]->id);

                    description.chunkId = timeline[i]->chunkId;
                    description.numberOfSamples = 1;
                    m_timeline.push_back(description);

                    auto sq = sequenceref(description.chunkId, i, k);
                    sq.numframes = 1;
                    m_sequences.push_back(sq);
                }
            }
            else
            {
                assert(false);
                SequenceDescription description;
                description.id = timeline[i]->id;
                description.chunkId = timeline[i]->chunkId;
                description.numberOfSamples = timeline[i]->numberOfSamples;
                m_timeline.push_back(description);

                m_sequenceIdToFeatureId.push_back(timeline[i]->id);

                auto sq = sequenceref(description.chunkId, i, 0);
                sq.numframes = description.numberOfSamples;
                m_sequences.push_back(sq);
            }
        }
    }

    bool BundlerSplitted::OldRequireChunk(size_t chunkindex)
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
                    std::unordered_map<std::string, size_t> empty;
                    msra::dbn::latticesource lattices(
                        std::pair<std::vector<std::wstring>, std::vector<std::wstring>>(),
                        empty);
                    chunkdata.requiredata(m_featkind[m], m_featdim[m], m_sampperiod[m], lattices, m_verbosity);
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

    bool BundlerSplitted::RequireChunk(size_t chunkindex)
    {
        bool result = false;
        for (const auto& d: m_featureDeserializers)
        {
            result |= d->RequireChunk(chunkindex);
        }

        return result;
    }

    void BundlerSplitted::ReleaseChunk(size_t chunkIndex)
    {
        for (const auto& d : m_featureDeserializers)
        {
            d->ReleaseChunk(chunkIndex);
        }
    }

    void BundlerSplitted::OldReleaseChunk(size_t chunkIndex)
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

    const Timeline& BundlerSplitted::GetTimeline() const
    {
        return m_timeline;
    }

    std::vector<InputDescriptionPtr> BundlerSplitted::GetInputs() const
    {
        return m_inputs;
    }

    SequenceData BundlerSplitted::OldGetSequenceById(size_t id)
    {
        SequenceData result;

        std::vector<msra::dbn::matrix> feat;              // buffer for holding curernt minibatch's frames
        std::vector<std::vector<size_t>> uids;               // buffer for storing current minibatch's frame-level label sequence

        auto_timer timergetbatch;
        assert(m_totalframes > 0);

        const size_t numStreams = m_allchunks.size();

        const std::vector<char> noboundaryflags;    // dummy

        const size_t spos = id; // positer->second;
        const size_t epos = spos + 1;

        // Note that the above loop loops over all chunks incl. those that we already should have.
        // This has an effect, e.g., if 'numsubsets' has changed (we will fill gaps).

        // determine the true #frames we return, for allocation--it is less than mbframes in the case of MPI/data-parallel sub-set mode
        size_t tspos = 0;
        for (size_t pos = spos; pos < epos; pos++)
        {
            tspos += m_timeline[id].numberOfSamples;
        }

        if (tspos == 0)
        {
            return result;
        }

        // resize feat and uids
        // eldak:s should return phone boundaries and sentendmark lattices transcripts etc.
        feat.resize(m_featureIndices.size()); // TODO numFeatureStreams
        uids.resize(m_classids.size()); // TODO numLabelStreams

        // TODO go to virtual stream input InputDescriptionPtr GetInput() const override;

        foreach_index(i, feat)
        {
            feat[i].resize(m_inputs[m_featureIndices[i]]->sampleLayout->GetDim(0), tspos);

            if (i == 0)
            {
                foreach_index(j, uids)
                {
                    if (issupervised())             // empty means unsupervised training -> return empty uids
                    {
                        uids[j].resize(tspos);
                    }
                    else
                    {
                        uids[i].clear();
                    }
                }
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

            size_t n = 0;
            for (size_t i = 0; i < numStreams; ++i)
            {
                const auto & chunkdata = m_allchunks[i][uttref.chunkindex];
                size_t dimension = m_inputs[m_featureIndices[i]]->sampleLayout->GetDim(0);

                // eldak - does it mean we have read the who?
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
                        leftextent = rightextent = msra::dbn::augmentationextent(uttframevectors[t].size(), dimension);
                    }
                    else
                    {
                        leftextent = m_leftcontext[i];
                        rightextent = m_rightcontext[i];
                    }

                    msra::dbn::augmentneighbors(uttframevectors, noboundaryflags, uttref.frameindex + t, leftextent, rightextent, feat[i], t + tspos);
                }

                // copy the frames and class labels
                if (i == 0)
                {
                    auto uttclassids = GetClassIds(uttref);
                    foreach_index(j, uttclassids)
                    {
                        for (size_t t = 0; t < sequence.numberOfSamples; t++)          // t = time index into source utterance
                        {
                            if (issupervised())
                            {
                                uids[j][t + tspos] = uttclassids[j][uttref.frameindex + t];
                            }
                        }
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

        for (auto it = m_featureIndices.begin(); it != m_featureIndices.end(); ++it)
        {
            Sequence r;
            size_t id = *it;

            const msra::dbn::matrixstripe featOri = msra::dbn::matrixstripe(feat[id], 0, feat[0].cols());
            const size_t dimensions = featOri.rows();
            const void* tmp = &featOri(0, 0);

            r.numberOfSamples = 1;

            // eldak: this should not be allocated each time.
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

            result.m_data.insert(std::make_pair(m_inputs[*it]->id, r));
        }

        for (size_t l = 0; l < m_labelIndices.size(); ++l)
        {
            Sequence r;
            size_t id = l;

            auto dimension = m_inputs[m_labelIndices[l]]->sampleLayout->GetDims()[0];
            size_t dim = dimension;

            const std::vector<size_t>& x = uids[id];

            if (m_elementSize == sizeof(float))
            {
                float* tmp = new float[dim];
                memset(tmp, 0, m_elementSize * dim);
                tmp[x[0]] = 1;
                r.data = tmp;
                r.numberOfSamples = 1;
            }
            else
            {
                double* tmp = new double[dim];
                tmp[x[0]] = 1;
                r.data = tmp;
                r.numberOfSamples = 1;
            }
            result.m_data.insert(std::make_pair(m_inputs[m_labelIndices[l]]->id, r));
        }

        return result;
    }

    SequenceData BundlerSplitted::GetSequenceById(size_t id)
    {
        assert(m_framemode);
        assert(m_featureDeserializers.size() == 1);
        assert(m_labelDeserializers.size() == 1);

        size_t originalSequenceId = m_sequenceIdToFeatureId[id];
        size_t orginalIndex = m_sequences[id].frameindex;
        size_t originalLabelId = m_sequenceIdTolabelId[0][originalSequenceId];

        Sequence f = m_featureDeserializers[0]->GetSampleById(originalSequenceId, orginalIndex);
        Sequence l = m_labelDeserializers[0]->GetSampleById(originalLabelId, orginalIndex);

        SequenceData result;
        result.m_data.insert(std::make_pair(m_inputs[m_featureIndices[0]]->id, f));
        result.m_data.insert(std::make_pair(m_inputs[m_labelIndices[0]]->id, l));

        return result;
    }

    void BundlerSplitted::SetEpochConfiguration(const EpochConfiguration& /*config*/)
    {
        // TODO do we keep SetEpochConfiguration(), now empty?
    }

    std::vector<BundlerSplitted::shiftedvector<msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE>>> BundlerSplitted::GetClassIds(
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
}}}
