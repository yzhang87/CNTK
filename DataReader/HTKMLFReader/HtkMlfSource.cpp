//
// <copyright file="HTKMLFSource.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFSource.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#ifdef _WIN32
#include <objbase.h>
#endif
#include "Basics.h"

#include "htkfeatio.h"                  // for reading HTK features
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "simplesenonehmm.h"            // for MMI scoring
#include "msra_mgram.h"                 // for unigram scores of ground-truth path in sequence training

#include "rollingwindowsource.h"        // minibatch sources
#include "utterancesourcemulti.h"
#include "chunkevalsource.h"
#include "minibatchiterator.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "commandArgUtil.h"
#include "HTKMLFSource.h"
#include "TimerUtility.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

#ifdef __unix__
#include <limits.h>
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned int UNINT32;
#endif
#pragma warning (disable: 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this

#include "HTKMLFSource.h"
#include "Utils.h"

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

            template<class ElemType>
            void HTKMLFSource<ElemType>::Init(const ConfigParameters& readerConfig)
            {
                ConfigArray numberOfuttsPerMinibatchForAllEpochs = readerConfig("nbruttsineachrecurrentiter", "1");

                vector<wstring> scriptpaths;
                vector<wstring> RootPathInScripts;
                vector<wstring> mlfpaths;
                vector<vector<wstring>>mlfpathsmulti;
                size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
                vector<vector<wstring>> infilesmulti;
                size_t numFiles;
                wstring unigrampath(L"");

                size_t iFeat, iLabel;
                iFeat = iLabel = 0;
                vector<wstring> statelistpaths;
                vector<size_t> numContextLeft;
                vector<size_t> numContextRight;

                std::vector<std::wstring> featureNames;
                std::vector<std::wstring> labelNames;

                // for hmm and lattice 
                std::vector<std::wstring> hmmNames;
                std::vector<std::wstring> latticeNames;
                GetDataNamesFromConfig(readerConfig, featureNames, labelNames, hmmNames, latticeNames);
                if (featureNames.size() + labelNames.size() <= 1)
                {
                    RuntimeError("network needs at least 1 input and 1 output specified!");
                }

                //load data for all real-valued inputs (features)
                foreach_index(i, featureNames)
                {
                    ConfigParameters thisFeature = readerConfig(featureNames[i]);
                    m_featDims.push_back(thisFeature("dim"));
                    ConfigArray contextWindow = thisFeature("contextWindow", "1");
                    if (contextWindow.size() == 1) // symmetric
                    {
                        size_t windowFrames = contextWindow[0];
                        if (windowFrames % 2 == 0)
                            RuntimeError("augmentationextent: neighbor expansion of input features to %d not symmetrical", windowFrames);
                        size_t context = windowFrames / 2;           // extend each side by this
                        numContextLeft.push_back(context);
                        numContextRight.push_back(context);

                    }
                    else if (contextWindow.size() == 2) // left context, right context
                    {
                        numContextLeft.push_back(contextWindow[0]);
                        numContextRight.push_back(contextWindow[1]);
                    }
                    else
                    {
                        RuntimeError("contextFrames must have 1 or 2 values specified, found %d", contextWindow.size());
                    }
                    // update m_featDims to reflect the total input dimension (featDim x contextWindow), not the native feature dimension
                    // that is what the lower level feature readers expect
                    m_featDims[i] = m_featDims[i] * (1 + numContextLeft[i] + numContextRight[i]);

                    string type = thisFeature("type", "Real");
                    if (type == "Real"){
                        m_nameToTypeMap[featureNames[i]] = InputOutputTypes::real;
                    }
                    else{
                        RuntimeError("feature type must be Real");
                    }

                    m_featureNameToIdMap[featureNames[i]] = iFeat;
                    scriptpaths.push_back(thisFeature("scpFile"));
                    RootPathInScripts.push_back(thisFeature("PrefixPathInSCP", ""));
                    m_featureNameToDimMap[featureNames[i]] = m_featDims[i];

                    InputDefinition input;
                    input.name = featureNames[i];
                    input.id = m_inputs.size();
                    input.dimensions.push_back(m_featDims[i]);
                    input.elementSize = sizeof(ElemType);
                    m_inputs.push_back(input);

                    iFeat++;
                }

                foreach_index(i, labelNames)
                {
                    ConfigParameters thisLabel = readerConfig(labelNames[i]);
                    if (thisLabel.Exists("labelDim"))
                        m_labelDims.push_back(thisLabel("labelDim"));
                    else if (thisLabel.Exists("dim"))
                        m_labelDims.push_back(thisLabel("dim"));
                    else
                        RuntimeError("labels must specify dim or labelDim");

                    string type;
                    if (thisLabel.Exists("labelType"))
                        type = thisLabel("labelType"); // let's deprecate this eventually and just use "type"...
                    else
                        type = thisLabel("type", "Category"); // outputs should default to category

                    if (type == "Category")
                        m_nameToTypeMap[labelNames[i]] = InputOutputTypes::category;
                    else
                        RuntimeError("label type must be Category");

                    statelistpaths.push_back(thisLabel("labelMappingFile", L""));

                    m_labelNameToIdMap[labelNames[i]] = iLabel;
                    m_labelNameToDimMap[labelNames[i]] = m_labelDims[i];
                    mlfpaths.clear();
                    if (thisLabel.ExistsCurrent("mlfFile"))
                    {
                        mlfpaths.push_back(thisLabel("mlfFile"));
                    }
                    else
                    {
                        if (!thisLabel.ExistsCurrent("mlfFileList"))
                        {
                            RuntimeError("Either mlfFile or mlfFileList must exist in HTKMLFReder");
                        }
                        wstring list = thisLabel("mlfFileList");
                        for (msra::files::textreader r(list); r;)
                        {
                            mlfpaths.push_back(r.wgetline());
                        }
                    }
                    mlfpathsmulti.push_back(mlfpaths);

                    InputDefinition input;
                    input.name = labelNames[i];
                    input.id = m_inputs.size();
                    input.dimensions.push_back(m_labelDims[i]);
                    input.elementSize = sizeof(ElemType);
                    m_inputs.push_back(input);

                    iLabel++;
                }

                //get lattice toc file names 
                std::pair<std::vector<wstring>, std::vector<wstring>> latticetocs;
                foreach_index(i, latticeNames)
                    //only support one set of lattice now
                {
                    ConfigParameters thislattice = readerConfig(latticeNames[i]);


                    vector<wstring> paths;
                    expand_wildcards(thislattice("denlatTocFile"), paths);
                    latticetocs.second.insert(latticetocs.second.end(), paths.begin(), paths.end());

                    if (thislattice.Exists("numlatTocFile"))
                    {
                        paths.clear();
                        expand_wildcards(thislattice("numlatTocFile"), paths);
                        latticetocs.first.insert(latticetocs.first.end(), paths.begin(), paths.end());
                    }

                    InputDefinition input;
                    input.name = latticeNames[i];
                    input.id = m_inputs.size();
                    input.dimensions.push_back(1);
                    input.elementSize = 0;
                    m_inputs.push_back(input);
                }

                //get HMM related file names
                vector<wstring> cdphonetyingpaths, transPspaths;
                foreach_index(i, hmmNames)
                {
                    ConfigParameters thishmm = readerConfig(hmmNames[i]);

                    vector<wstring> paths;
                    cdphonetyingpaths.push_back(thishmm("phoneFile"));
                    transPspaths.push_back(thishmm("transpFile", L""));
                }

                // mmf files 
                //only support one set now
                if (cdphonetyingpaths.size() > 0 && statelistpaths.size() > 0 && transPspaths.size() > 0)
                    m_hset.loadfromfile(cdphonetyingpaths[0], statelistpaths[0], transPspaths[0]);
                if (iFeat != scriptpaths.size() || iLabel != mlfpathsmulti.size())
                    RuntimeError(msra::strfun::strprintf("# of inputs files vs. # of inputs or # of output files vs # of outputs inconsistent\n"));

                m_frameMode = readerConfig("frameMode", "true");
                m_verbosity = readerConfig("verbosity", "2");

                // determine if we partial minibatches are desired
                std::string minibatchMode(readerConfig("minibatchMode", "Partial"));
                m_partialMinibatch = !_stricmp(minibatchMode.c_str(), "Partial");

                // read all input files (from multiple inputs)
                // TO DO: check for consistency (same number of files in each script file)
                numFiles = 0;
                foreach_index(i, scriptpaths)
                {
                    vector<wstring> filelist;
                    std::wstring scriptpath = scriptpaths[i];
                    fprintf(stderr, "reading script file %S ...", scriptpath.c_str());
                    size_t n = 0;
                    for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly/*optimization*/;)
                    {
                        filelist.push_back(reader.wgetline());
                        n++;
                    }

                    fprintf(stderr, " %lu entries\n", n);

                    if (i == 0)
                        numFiles = n;
                    else
                        if (n != numFiles)
                            RuntimeError(msra::strfun::strprintf("number of files in each scriptfile inconsistent (%d vs. %d)", numFiles, n));

                    // post processing file list : 
                    //  - if users specified PrefixPath, add the prefix to each of path in filelist
                    //  - else do the dotdotdot expansion if necessary 
                    wstring rootpath = RootPathInScripts[i];
                    if (!rootpath.empty()) // use has specified a path prefix for this  feature 
                    {
                        // first make slash consistent (sorry for linux users:this is not necessary for you)
                        std::replace(rootpath.begin(), rootpath.end(), L'\\', L'/');
                        // second, remove trailling slash if there is any 
                        std::wregex trailer(L"/+$");
                        rootpath = std::regex_replace(rootpath, trailer, wstring(L""));
                        // third, join the rootpath with each entry in filelist 
                        if (!rootpath.empty())
                        {
                            for (wstring & path : filelist)
                            {
                                if (path.find_first_of(L'=') != wstring::npos)
                                {
                                    vector<wstring> strarr = msra::strfun::split(path, L"=");
#ifdef WIN32
                                    replace(strarr[1].begin(), strarr[1].end(), L'\\', L'/');
#endif 

                                    path = strarr[0] + L"=" + rootpath + L"/" + strarr[1];
                                }
                                else
                                {
#ifdef WIN32
                                    replace(path.begin(), path.end(), L'\\', L'/');
#endif 
                                    path = rootpath + L"/" + path;
                                }
                            }
                        }
                    }
                    else
                    {
                        /*
                        do "..." expansion if SCP uses relative path names
                        "..." in the SCP means full path is the same as the SCP file
                        for example, if scp file is "//aaa/bbb/ccc/ddd.scp"
                        and contains entry like
                        .../file1.feat
                        .../file2.feat
                        etc.
                        the features will be read from
                        //aaa/bbb/ccc/file1.feat
                        //aaa/bbb/ccc/file2.feat
                        etc.
                        This works well if you store the scp file with the features but
                        do not want different scp files everytime you move or create new features
                        */
                        wstring scpdircached;
                        for (auto & entry : filelist)
                            ExpandDotDotDot(entry, scriptpath, scpdircached);
                    }


                    infilesmulti.push_back(std::move(filelist));
                }

                if (readerConfig.Exists("unigram"))
                    unigrampath = (wstring)readerConfig("unigram");

                // load a unigram if needed (this is used for MMI training)
                msra::lm::CSymbolSet unigramsymbols;
                std::unique_ptr<msra::lm::CMGramLM> unigram;
                if (unigrampath != L"")
                {
                    unigram.reset(new msra::lm::CMGramLM());
                    unigram->read(unigrampath, unigramsymbols, false/*filterVocabulary--false will build the symbol map*/, 1/*maxM--unigram only*/);
                }

                if (!unigram)
                    fprintf(stderr, "trainlayer: OOV-exclusion code enabled, but no unigram specified to derive the word set from, so you won't get OOV exclusion\n");

                // currently assumes all mlfs will have same root name (key)
                set<wstring> restrictmlftokeys;     // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
                if (infilesmulti[0].size() <= 100)
                {
                    foreach_index(i, infilesmulti[0])
                    {
                        msra::asr::htkfeatreader::parsedpath ppath(infilesmulti[0][i]);
                        const wstring key = regex_replace((wstring)ppath, wregex(L"\\.[^\\.\\\\/:]*$"), wstring());  // delete extension (or not if none)
                        restrictmlftokeys.insert(key);
                    }
                }
                // get labels

                double htktimetoframe = 100000.0;           // default is 10ms 
                std::vector<std::map<std::wstring, std::vector<msra::asr::htkmlfentry>>> labelsmulti;
                foreach_index(i, mlfpathsmulti)
                {
                    const msra::lm::CSymbolSet* wordmap = unigram ? &unigramsymbols : NULL;
                    msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>
                        labels(mlfpathsmulti[i], restrictmlftokeys, statelistpaths[i], wordmap, (map<string, size_t>*) NULL, htktimetoframe);      // label MLF
                    // get the temp file name for the page file

                    // Make sure 'msra::asr::HTKMLFSource' type has a move constructor
                    static_assert(std::is_move_constructible<msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>>::value,
                        "Type 'msra::asr::HTKMLFSource' should be move constructible!");

                    labelsmulti.push_back(std::move(labels));
                }

                // construct all the parameters we don't need, but need to be passed to the constructor...
                m_lattices.reset(new msra::dbn::latticesource(latticetocs, m_hset.getsymmap()));

                numContextLeft.push_back(0);
                numContextRight.push_back(0);
                // ALways "block randomize" source.
                // now get the frame source. This has better randomization and doesn't create temp files
                m_frameSource.reset(new msra::dbn::minibatchutterancesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, numContextLeft, numContextRight, randomizeNone, *m_lattices, m_latticeMap, m_frameMode));
                m_frameSource->setverbosity(m_verbosity);
            }

            //StartMinibatchLoop - Startup a minibatch loop 
            // requestedMBSize - [in] size of the minibatch (number of frames, etc.)
            // epoch - [in] epoch number for this loop
            // requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
            template<class ElemType>
            void HTKMLFSource<ElemType>::StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /*= requestDataSize*/)
            {
                assert(subsetNum < numSubsets);
                assert(((subsetNum == 0) && (numSubsets == 1)) || this->SupportsDistributedMBRead());

                m_mbNumTimeSteps = requestedMBSize;       // note: ignored in frame mode and full-sequence mode

                // BUGBUG: in BPTT and sequence mode, we should pass 1 or 2 instead of requestedMBSize to ensure we only get one utterance back at a time
                StartMinibatchLoopToTrainOrTest(requestedMBSize, epoch, subsetNum, numSubsets, requestedEpochSamples);

            }

            template<class ElemType>
            void HTKMLFSource<ElemType>::StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples)
            {
                size_t totalFrames = m_frameSource->totalframes();

                size_t extraFrames = totalFrames%mbSize;
                size_t minibatches = totalFrames / mbSize;

                // if we are allowing partial minibatches, do nothing, and let it go through
                if (!m_partialMinibatch)
                {
                    // we don't want any partial frames, so round total frames to be an even multiple of our mbSize
                    if (totalFrames > mbSize)
                        totalFrames -= extraFrames;

                    if (requestedEpochSamples == requestDataSize)
                    {
                        requestedEpochSamples = totalFrames;
                    }
                    else if (minibatches > 0)   // if we have any full minibatches
                    {
                        // since we skip the extraFrames, we need to add them to the total to get the actual number of frames requested
                        size_t sweeps = (requestedEpochSamples - 1) / totalFrames; // want the number of sweeps we will skip the extra, so subtract 1 and divide
                        requestedEpochSamples += extraFrames*sweeps;
                    }
                }
                else if (requestedEpochSamples == requestDataSize)
                {
                    requestedEpochSamples = totalFrames;
                }

                m_mbiter.reset(new msra::dbn::minibatchiterator(*m_frameSource, epoch, requestedEpochSamples, mbSize, subsetNum, numSubsets, 1));
                AdvanceIteratorToNextDataPortion();

                m_noData = false;
                if (!*m_mbiter)
                    m_noData = true;
            }

            template<class ElemType>
            bool HTKMLFSource<ElemType>::GetMinibatch4SEToTrainOrTest(std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>> & latticeinput,
                std::vector<size_t> &uids, std::vector<size_t> &boundaries, std::vector<size_t> &extrauttmap)
            {
                latticeinput.clear();
                uids.clear();
                boundaries.clear();
                extrauttmap.clear();
                for (size_t i = 0; i < m_extraSeqsPerMB.size(); i++)
                {
                    latticeinput.push_back(m_extraLatticeBufferMultiUtt[i]);
                    uids.insert(uids.end(), m_extraLabelsIDBufferMultiUtt[i].begin(), m_extraLabelsIDBufferMultiUtt[i].end());
                    boundaries.insert(boundaries.end(), m_extraPhoneboundaryIDBufferMultiUtt[i].begin(), m_extraPhoneboundaryIDBufferMultiUtt[i].end());
                }

                extrauttmap.insert(extrauttmap.end(), m_extraSeqsPerMB.begin(), m_extraSeqsPerMB.end());
                return true;
            }

            template<class ElemType>
            bool HTKMLFSource<ElemType>::GetHmmData(msra::asr::simplesenonehmm * hmm)
            {
                *hmm = m_hset;
                return true;
            }

            template<class ElemType>
            Timeline& HTKMLFSource<ElemType>::getTimeline()
            {
                throw std::logic_error("The method or operation is not implemented.");
            }

            template<class ElemType>
            std::map<size_t, Sequence> HTKMLFSource<ElemType>::getSequenceById(sequenceId /*id*/)
            {
                // Currently get the next sequence.
                AdvanceIteratorToNextDataPortion();
                if (!*m_mbiter)
                {
                    return std::map<size_t, Sequence>();
                }

                std::map<size_t, Sequence> result;

                size_t i = 0;
                for (; i < m_featDims.size(); ++i)
                {
                    const msra::dbn::matrixstripe featOri = m_mbiter->frames(i);
                    const size_t actualmbsizeOri = featOri.cols();

                    Sequence s;
                    for (int k = 0; k < actualmbsizeOri; k++) // column major, so iterate columns in outside loop
                    {
                        Frame f;
                        for (int d = 0; d < featOri.rows(); d++)
                        {
                            auto l = featOri(d, k);
                            f.features.resize(sizeof(l));
                            memcpy(&f.features[0], &l, sizeof(l));
                        }
                        s.frames.push_back(f);
                    }

                    result.insert(std::make_pair(i, s));
                }

                //for (auto it = m_labelNameToIdMap.begin(); it != m_labelNameToIdMap.end(); ++it)
                //{
                //    size_t id = m_labelNameToIdMap[it->first];
                //    size_t dim = m_labelNameToDimMap[it->first];

                //    const vector<size_t> & uids = m_mbiter->labels(id);
                //    size_t actualmbsizeOri = uids.size();

                //    // loop through the columns and set one value to 1
                //    // in the future we want to use a sparse matrix here
                //    for (int k = 0; k < actualmbsizeOri; k++)
                //    {
                //        assert(uids[k] < dim);
                //        m_labelsBufferMultiUtt[i].get()[k*dim + uids[k] + m_labelsStartIndexMultiUtt[id + i*numOfLabel]] = (ElemType)1;
                //    }

                //    result

                //    i++;
                //}
                ////lattice
                //if (m_latticeBufferMultiUtt[i] != NULL)
                //{
                //    m_latticeBufferMultiUtt[i].reset();
                //}

                //if (m_mbiter->haslattice())
                //{
                //    m_latticeBufferMultiUtt[i] = std::move(m_mbiter->lattice(0));
                //    m_phoneboundaryIDBufferMultiUtt[i].clear();
                //    m_phoneboundaryIDBufferMultiUtt[i] = m_mbiter->bounds();
                //    m_labelsIDBufferMultiUtt[i].clear();
                //    m_labelsIDBufferMultiUtt[i] = m_mbiter->labels();
                //}

                return result;
            }

            template<class ElemType>
            void HTKMLFSource<ElemType>::AdvanceIteratorToNextDataPortion()
            {
                // Advance the MB iterator until we find some data or reach the end of epoch
                while ((m_mbiter->currentmbframes() == 0) && *m_mbiter)
                {
                    (*m_mbiter)++;
                }
            }

            template<class ElemType>
            std::vector<InputDefinition> HTKMLFSource<ElemType>::getInputs()
            {
                return m_inputs;
            }


            template class HTKMLFSource<float>;
            template class HTKMLFSource<double>;
        }
    }
}
