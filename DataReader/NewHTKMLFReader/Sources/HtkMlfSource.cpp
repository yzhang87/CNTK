#include "stdafx.h"
#include <regex>
#include <set>
#include "HtkMlfSource.h"
#include "msra_mgram.h"
#include "htkfeatio.h"
#include "utterancesourcemulti.h"
#include "rollingwindowsource.h"

static void ExpandDotDotDot(wstring & featPath, const wstring & scpPath, wstring & scpDirCached)
{
    wstring delim = L"/\\";

    if (scpDirCached.empty())
    {
        scpDirCached = scpPath;
        wstring tail;
        auto pos = scpDirCached.find_last_of(delim);
        if (pos != wstring::npos)
        {
            tail = scpDirCached.substr(pos + 1);
            scpDirCached.resize(pos);
        }
        if (tail.empty()) // nothing was split off: no dir given, 'dir' contains the filename
            scpDirCached.swap(tail);
    }
    size_t pos = featPath.find(L"...");
    if (pos != featPath.npos)
        featPath = featPath.substr(0, pos) + scpDirCached + featPath.substr(pos + 3);
}

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            // Load all input and output data. 
            // Note that the terms features imply be real-valued quanities and
            // labels imply categorical quantities, irrespective of whether they
            // are inputs or targets for the network
            template<class ElemType>
            HTKMLFSource<ElemType>::HTKMLFSource(const ConfigParameters& readerConfig)
            {
                vector<wstring> scriptpaths;
                vector<wstring> RootPathInScripts;
                vector<wstring> mlfpaths;
                vector<vector<wstring>>mlfpathsmulti;

                size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
                vector<vector<wstring>> infilesmulti;
                size_t numFiles;
                wstring unigrampath(L"");

                //wstring statelistpath(L"");
                size_t randomize = randomizeAuto;
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

                    // eldak: Should be moved to augmentation.
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

                    m_featuresBufferMultiIO.push_back(nullptr);
                    m_featuresBufferAllocatedMultiIO.push_back(0);

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

                    m_labelsBufferMultiIO.push_back(nullptr);
                    m_labelsBufferAllocatedMultiIO.push_back(0);

                    iLabel++;

                    wstring labelToTargetMappingFile(thisLabel("labelToTargetMappingFile", L""));
                    if (labelToTargetMappingFile != L"")
                    {
                        std::vector<std::vector<ElemType>> labelToTargetMap;
                        m_convertLabelsToTargetsMultiIO.push_back(true);
                        if (thisLabel.Exists("targetDim"))
                        {
                            m_labelNameToDimMap[labelNames[i]] = m_labelDims[i] = thisLabel("targetDim");
                        }
                        else
                            RuntimeError("output must specify targetDim if labelToTargetMappingFile specified!");
                        size_t targetDim = ReadLabelToTargetMappingFile(labelToTargetMappingFile, statelistpaths[i], labelToTargetMap);
                        if (targetDim != m_labelDims[i])
                            RuntimeError("mismatch between targetDim and dim found in labelToTargetMappingFile");
                        m_labelToTargetMapMultiIO.push_back(labelToTargetMap);
                    }
                    else
                    {
                        m_convertLabelsToTargetsMultiIO.push_back(false);
                        m_labelToTargetMapMultiIO.push_back(std::vector<std::vector<ElemType>>());
                    }
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


                // eldak: Does not belong here, should be moved to randomizer.
                if (readerConfig.Exists("randomize"))
                {
                    const std::string& randomizeString = readerConfig("randomize");
                    if (randomizeString == "None")
                    {
                        randomize = randomizeNone;
                    }
                    else if (randomizeString == "Auto")
                    {
                        randomize = randomizeAuto;
                    }
                    else
                    {
                        randomize = readerConfig("randomize");
                    }
                }

                m_frameMode = readerConfig("frameMode", "true");
                m_verbosity = readerConfig("verbosity", "2");

                // eldak: should be moved to the packer.
                // determine if we partial minibatches are desired
                std::string minibatchMode(readerConfig("minibatchMode", "Partial"));
                m_partialMinibatch = !_stricmp(minibatchMode.c_str(), "Partial");

                // eldak: should be moved to the randomizer.
                // get the read method, defaults to "blockRandomize" other option is "rollingWindow"
                std::string readMethod(readerConfig("readMethod", "blockRandomize"));

                if (readMethod == "blockRandomize" && randomize == randomizeNone)
                {
                    fprintf(stderr, "WARNING: Randomize cannot be set to None when readMethod is set to blockRandomize. Change it Auto");
                    randomize = randomizeAuto;
                }

                // eldak: should user the block reader interface only.
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
                        {

                            ExpandDotDotDot(entry, scriptpath, scpdircached);
                        }
                    }


                    infilesmulti.push_back(std::move(filelist));
                }

                if (readerConfig.Exists("unigram"))
                    unigrampath = (wstring)readerConfig("unigram");

                // load a unigram if needed (this is used for MMI training)
                msra::lm::CSymbolSet unigramsymbols;
                std::unique_ptr<msra::lm::CMGramLM> unigram;
                size_t silencewordid = SIZE_MAX;
                size_t startwordid = SIZE_MAX;
                size_t endwordid = SIZE_MAX;
                if (unigrampath != L"")
                {
                    unigram.reset(new msra::lm::CMGramLM());
                    unigram->read(unigrampath, unigramsymbols, false/*filterVocabulary--false will build the symbol map*/, 1/*maxM--unigram only*/);
                    silencewordid = unigramsymbols["!silence"];     // give this an id (even if not in the LM vocabulary)
                    startwordid = unigramsymbols["<s>"];
                    endwordid = unigramsymbols["</s>"];
                }

                if (!unigram)
                    fprintf(stderr, "trainlayer: OOV-exclusion code enabled, but no unigram specified to derive the word set from, so you won't get OOV exclusion\n");

                // currently assumes all mlfs will have same root name (key)
                std::set<wstring> restrictmlftokeys;     // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
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

                //if (readerConfig.Exists("statelist"))
                //    statelistpath = readerConfig("statelist");

                double htktimetoframe = 100000.0;           // default is 10ms 
                //std::vector<msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>> labelsmulti;
                std::vector<std::map<std::wstring, std::vector<msra::asr::htkmlfentry>>> labelsmulti;
                //std::vector<std::wstring> pagepath;
                foreach_index(i, mlfpathsmulti)
                {
                    const msra::lm::CSymbolSet* wordmap = unigram ? &unigramsymbols : NULL;
                    msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>
                        labels(mlfpathsmulti[i], restrictmlftokeys, statelistpaths[i], wordmap, (map<string, size_t>*) NULL, htktimetoframe);      // label MLF
                    // get the temp file name for the page file

                    // Make sure 'msra::asr::htkmlfreader' type has a move constructor
                    static_assert(std::is_move_constructible<msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>>::value,
                        "Type 'msra::asr::htkmlfreader' should be move constructible!");

                    labelsmulti.push_back(std::move(labels));
                }

                // eldak: should be moved to randomizer.
                if (!_stricmp(readMethod.c_str(), "blockRandomize"))
                {
                    // construct all the parameters we don't need, but need to be passed to the constructor...
                    m_lattices.reset(new msra::dbn::latticesource(latticetocs, m_hset.getsymmap()));

                    // now get the frame source. This has better randomization and doesn't create temp files
                    m_frameSource.reset(new msra::dbn::minibatchutterancesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, numContextLeft, numContextRight, randomize, *m_lattices, m_latticeMap, m_frameMode));
                    m_frameSource->setverbosity(m_verbosity);
                }
                else if (!_stricmp(readMethod.c_str(), "rollingWindow"))
                {
#ifdef _WIN32
                    std::wstring pageFilePath;
#else
                    std::string pageFilePath;
#endif
                    std::vector<std::wstring> pagePaths;
                    if (readerConfig.Exists("pageFilePath"))
                    {
                        pageFilePath = readerConfig("pageFilePath");

                        // replace any '/' with '\' for compat with default path
                        std::replace(pageFilePath.begin(), pageFilePath.end(), '/', '\\');
#ifdef _WIN32               
                        // verify path exists
                        DWORD attrib = GetFileAttributes(pageFilePath.c_str());
                        if (attrib == INVALID_FILE_ATTRIBUTES || !(attrib & FILE_ATTRIBUTE_DIRECTORY))
                            RuntimeError("pageFilePath does not exist");
#endif
#ifdef __unix__
                        struct stat statbuf;
                        if (stat(pageFilePath.c_str(), &statbuf) == -1)
                        {
                            RuntimeError("pageFilePath does not exist");
                        }

#endif
                    }
                    else  // using default temporary path
                    {
#ifdef _WIN32
                        pageFilePath.reserve(MAX_PATH);
                        GetTempPath(MAX_PATH, &pageFilePath[0]);
#endif
#ifdef __unix__
                        pageFilePath.reserve(PATH_MAX);
                        pageFilePath = "/tmp/temp.CNTK.XXXXXX";
#endif
                    }

#ifdef _WIN32
                    if (pageFilePath.size() > MAX_PATH - 14) // max length of input to GetTempFileName is MAX_PATH-14
                        RuntimeError(msra::strfun::strprintf("pageFilePath must be less than %d characters", MAX_PATH - 14));
#endif
#ifdef __unix__
                    if (pageFilePath.size() > PATH_MAX - 14) // max length of input to GetTempFileName is PATH_MAX-14
                        RuntimeError(msra::strfun::strprintf("pageFilePath must be less than %d characters", PATH_MAX - 14));
#endif
                    foreach_index(i, infilesmulti)
                    {
#ifdef _WIN32
                        wchar_t tempFile[MAX_PATH];
                        GetTempFileName(pageFilePath.c_str(), L"CNTK", 0, tempFile);
                        pagePaths.push_back(tempFile);
#endif
#ifdef __unix__
                        char* tempFile;
                        //GetTempFileName(pageFilePath.c_str(), L"CNTK", 0, tempFile);
                        tempFile = (char*)pageFilePath.c_str();
                        int fid = mkstemp(tempFile);
                        unlink(tempFile);
                        close(fid);
                        pagePaths.push_back(GetWC(tempFile));
#endif
                    }

                    const bool mayhavenoframe = false;
                    int addEnergy = 0;

                    m_frameSource.reset(new msra::dbn::minibatchframesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, numContextLeft, numContextRight, randomize, pagePaths, mayhavenoframe, addEnergy));
                    m_frameSource->setverbosity(m_verbosity);
                }
                else
                {
                    RuntimeError("readMethod must be rollingWindow or blockRandomize");
                }
            }

            template<class ElemType>
            void HTKMLFSource<ElemType>::GetDataNamesFromConfig(
                const ConfigParameters& readerConfig,
                std::vector<std::wstring>& features,
                std::vector<std::wstring>& labels,
                std::vector<std::wstring>& hmms,
                std::vector<std::wstring>& lattices)
            {
                for (auto iter = readerConfig.begin(); iter != readerConfig.end(); ++iter)
                {
                    auto pair = *iter;
                    ConfigParameters temp = iter->second;
                    // see if we have a config parameters that contains a "file" element, it's a sub key, use it
                    if (temp.ExistsCurrent("scpFile"))
                    {
                        features.push_back(msra::strfun::utf16(iter->first));
                    }
                    else if (temp.ExistsCurrent("mlfFile") || temp.ExistsCurrent("mlfFileList"))
                    {
                        labels.push_back(msra::strfun::utf16(iter->first));
                    }
                    else if (temp.ExistsCurrent("phoneFile"))
                    {
                        hmms.push_back(msra::strfun::utf16(iter->first));
                    }
                    else if (temp.ExistsCurrent("denlatTocFile"))
                    {
                        lattices.push_back(msra::strfun::utf16(iter->first));
                    }
                }
            }

            template<class ElemType>
            size_t HTKMLFSource<ElemType>::ReadLabelToTargetMappingFile(const std::wstring& labelToTargetMappingFile, const std::wstring& labelListFile, std::vector<std::vector<ElemType>>& labelToTargetMap)
            {
                if (labelListFile == L"")
                    RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): cannot read labelToTargetMappingFile without a labelMappingFile!");

                vector<std::wstring> labelList;
                size_t count, numLabels;
                count = 0;
                // read statelist first
                msra::files::textreader labelReader(labelListFile);
                while (labelReader)
                {
                    labelList.push_back(labelReader.wgetline());
                    count++;
                }
                numLabels = count;
                count = 0;
                msra::files::textreader mapReader(labelToTargetMappingFile);
                size_t targetDim = 0;
                while (mapReader)
                {
                    std::wstring line(mapReader.wgetline());
                    // find white space as a demarcation
                    std::wstring::size_type pos = line.find(L" ");
                    std::wstring token = line.substr(0, pos);
                    std::wstring targetstring = line.substr(pos + 1);

                    if (labelList[count] != token)
                        RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): mismatch between labelMappingFile and labelToTargetMappingFile");

                    if (count == 0)
                        targetDim = targetstring.length();
                    else if (targetDim != targetstring.length())
                        RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): inconsistent target length among records");

                    std::vector<ElemType> targetVector(targetstring.length(), (ElemType)0.0);
                    foreach_index(i, targetstring)
                    {
                        if (targetstring.compare(i, 1, L"1") == 0)
                            targetVector[i] = (ElemType)1.0;
                        else if (targetstring.compare(i, 1, L"0") != 0)
                            RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): expecting label2target mapping to contain only 1's or 0's");
                    }
                    labelToTargetMap.push_back(targetVector);
                    count++;
                }

                // verify that statelist and label2target mapping file are in same order (to match up with reader) while reading mapping
                if (count != labelList.size())
                    RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): mismatch between lengths of labelMappingFile vs labelToTargetMappingFile");

                return targetDim;
            }

            template class HTKMLFSource<float>;
            template class HTKMLFSource<double>;
        }
    }
}