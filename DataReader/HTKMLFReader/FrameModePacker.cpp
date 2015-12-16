#include "stdafx.h"
#include "FrameModePacker.h"
#include "TimerUtility.h"
#include "Utils.h"
#include <DataReader.h>
#include "BundlerSplitted.h"
#include "ConfigHelper.h"
#include "BlockRandomizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    InputDescriptionPtr GetInputByName(const std::wstring& name, const std::vector<InputDescriptionPtr>& inputs)
    {
        auto predicate = [name](InputDescriptionPtr input) {
            return input->name == name;
        };

        auto input = std::find_if(std::begin(inputs), std::end(inputs), predicate);
        if (input == std::end(inputs))
        {
            InvalidArgument("Unknown feature!");
        }

        return *input;
    }

    FrameModePacker::EpochImplementation::EpochImplementation(FrameModePacker* parent)
        : m_parent(parent)
    {}

    FrameModePacker::EpochImplementation::~EpochImplementation()
    {}

    Minibatch FrameModePacker::EpochImplementation::ReadMinibatch()
    {
        return m_parent->GetMinibatch();
    }

    FrameModePacker::FrameModePacker(const ConfigParameters& config, MemoryProviderPtr memoryProvider, size_t elementSize)
        : m_pMBLayout(make_shared<MBLayout>())
        , m_memoryProvider(memoryProvider)
        , m_elementSize(elementSize)
    {
        InitFromConfig(config);
    }

    std::vector<InputDescriptionPtr> FrameModePacker::GetInputs()
    {
        return m_transformer->GetInputs();
    }

    EpochPtr FrameModePacker::StartNextEpoch(const EpochConfiguration& config)
    {
        assert(config.workerRank < config.numberOfWorkers);

        // TODO: what to do with partial minibatches? Is it important to propagate this information to lower layers?
        m_transformer->SetEpochConfiguration(config);

        StartDistributedMinibatchLoop(config.minibatchSize, config.index, config.workerRank, config.numberOfWorkers, config.totalSize);
        return std::make_unique<EpochImplementation>(this);
    }

    void FrameModePacker::InitFromConfig(const ConfigParameters & readerConfig)
    {
        size_t window = ConfigHelper::GetRandomizationWindow(readerConfig);

        auto deserializers = CreateDeserializers(readerConfig, true, m_elementSize);
        assert(deserializers.size() == 2);

        auto bundler = std::make_shared<BundlerSplitted>(readerConfig, true, m_verbosity, deserializers[0], deserializers);

        std::wstring readMethod = ConfigHelper::GetRandomizer(readerConfig);
        if (_wcsicmp(readMethod.c_str(), L"blockRandomize"))
        {
            RuntimeError("readMethod must be 'blockRandomize'");
        }
        m_transformer = std::make_shared<BlockRandomizer>(m_verbosity, window, bundler);

        intargvector numberOfuttsPerMinibatchForAllEpochs =
            readerConfig(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int>{ 1 })));
        Utils::CheckMinibatchSizes(numberOfuttsPerMinibatchForAllEpochs);

        m_numSeqsPerMBForAllEpochs = numberOfuttsPerMinibatchForAllEpochs;

        // (SGD will ask before entering actual reading --TODO: This is hacky.)
        m_numSeqsPerMB = m_numSeqsPerMBForAllEpochs[0];
        m_pMBLayout->Init(m_numSeqsPerMB, 0, true); 
        m_noData = false;

        if (readerConfig.Exists(L"legacyMode"))
            RuntimeError("legacy mode has been deprecated\n");

        // eldak: we should introduce a separate class describing inputs with proper interface.
        std::vector<InputDescriptionPtr> inputs = m_transformer->GetInputs();
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            m_nameToId.insert(std::make_pair(inputs[i]->name, inputs[i]->id));
        }

        size_t iFeat = 0, iLabel = 0;

        std::vector<std::wstring> featureNames;
        std::vector<std::wstring> labelNames;
        std::vector<std::wstring> notused;
        Utils::GetDataNamesFromConfig(readerConfig, featureNames, labelNames, notused, notused);

        // eldak: why not consolidate features and labels on this level? should not they be packed in the same manner essentially?
        foreach_index(i, featureNames)
        {
            const std::wstring& featureName = featureNames[i];
            auto input = GetInputByName(featureName, inputs);

            const ConfigParameters& thisFeature = readerConfig(featureName);
            m_featDims.push_back(input->sampleLayout->GetNumElements());

            wstring type = thisFeature(L"type", L"real");
            if (!_wcsicmp(type.c_str(), L"real"))
            {
                m_nameToTypeMap[featureName] = InputOutputTypes::real;
            }
            else
            {
                InvalidArgument("feature type must be 'real'");
            }

            m_featureNameToDimMap[featureName] = m_featDims[i];
            m_featureNameToIdMap[featureName] = iFeat;
            m_featuresBufferMultiIO.push_back(nullptr);
            m_featuresBufferAllocatedMultiIO.push_back(0);
            iFeat++;
        }

        foreach_index(i, labelNames)
        {
            const std::wstring& labelName = labelNames[i];
            auto input = GetInputByName(labelName, inputs);

            m_labelDims.push_back(input->sampleLayout->GetNumElements());

            const ConfigParameters& thisLabel = readerConfig(labelName);
            wstring type;
            if (thisLabel.Exists(L"labelType"))
                type = (const wstring &)thisLabel(L"labelType"); // let's deprecate this eventually and just use "type"...
            else
                type = (const wstring &)thisLabel(L"type", L"category"); // outputs should default to category

            if (!_wcsicmp(type.c_str(), L"category"))
                m_nameToTypeMap[labelNames[i]] = InputOutputTypes::category;
            else
                InvalidArgument("label type must be 'category'");

            m_labelNameToIdMap[labelNames[i]] = iLabel;
            m_labelNameToDimMap[labelNames[i]] = m_labelDims[i];
            m_labelsBufferMultiIO.push_back(nullptr);
            m_labelsBufferAllocatedMultiIO.push_back(0);
            iLabel++;
        }

        m_verbosity = readerConfig(L"verbosity", 2);

        // determine if we partial minibatches are desired
        wstring minibatchMode(readerConfig(L"minibatchMode", L"partial"));
        m_partialMinibatch = !_wcsicmp(minibatchMode.c_str(), L"partial");
    }

    //StartMinibatchLoop - Startup a minibatch loop 
    // requestedMBSize - [in] size of the minibatch (number of frames, etc.)
    // epoch - [in] epoch number for this loop
    // requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
    void FrameModePacker::StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t /*subsetNum*/, size_t /*numSubsets*/, size_t /*requestedEpochSamples = requestDataSize*/)
    {
        m_mbNumTimeSteps = requestedMBSize;
        m_numSeqsPerMB = m_numSeqsPerMBForAllEpochs[epoch];
        m_pMBLayout->Init(m_numSeqsPerMB, 0, false); // (SGD will ask before entering actual reading --TODO: This is hacky.)

        // resize the arrays
        // These are sized to the requested number. If not all can be filled, it will still return this many, just with gaps.
        // In frame mode, m_numSeqsPerMB must be 1. However, the returned layout has one 1-frame sequence per frame.
        m_numFramesToProcess.assign(m_numSeqsPerMB, 0);

        // for the multi-utterance process
        m_featuresBufferMultiUtt.assign(m_numSeqsPerMB, nullptr);
        m_featuresBufferAllocatedMultiUtt.assign(m_numSeqsPerMB, 0);
        m_labelsBufferMultiUtt.assign(m_numSeqsPerMB, nullptr);
        m_labelsBufferAllocatedMultiUtt.assign(m_numSeqsPerMB, 0);

        if ((m_numSeqsPerMB > 1))
        {
            LogicError("nbrUttsInEachRecurrentIter cannot be more than 1 in frame mode reading.");
        }

        m_noData = false;
        m_requestedMBSize = requestedMBSize;

        if (!m_featuresBufferMultiIO.empty())
        {
            if (m_featuresBufferMultiIO[0] != nullptr) // check first feature, if it isn't NULL, safe to assume all are not NULL? 
            {
                foreach_index(i, m_featuresBufferMultiIO)
                {
                    m_featuresBufferMultiIO[i] = nullptr;
                    m_featuresBufferAllocatedMultiIO[i] = 0;
                }
            }

            m_featuresStartIndexMultiUtt.assign(m_featuresBufferMultiIO.size()*m_numSeqsPerMB, 0);

        }

        if (!m_labelsBufferMultiIO.empty())
        {
            if (m_labelsBufferMultiIO[0] != nullptr)
            {
                foreach_index(i, m_labelsBufferMultiIO)
                {
                    m_labelsBufferMultiIO[i] = nullptr;
                    m_labelsBufferAllocatedMultiIO[i] = 0;
                }
            }

            m_labelsStartIndexMultiUtt.assign(m_labelsBufferMultiIO.size()*m_numSeqsPerMB, 0);
        }

        for (size_t u = 0; u < m_numSeqsPerMB; u++)
        {
            if (m_featuresBufferMultiUtt[u] != NULL)
            {
                m_featuresBufferMultiUtt[u] = NULL;
                m_featuresBufferAllocatedMultiUtt[u] = 0;
            }

            if (m_labelsBufferMultiUtt[u] != NULL)
            {
                m_labelsBufferMultiUtt[u] = NULL;
                m_labelsBufferAllocatedMultiUtt[u] = 0;
            }


            ReNewBufferForMultiIO(u);
        }
    }

    Minibatch FrameModePacker::GetMinibatch()
    {
        assert(m_numSeqsPerMB == 1);

        ScopeTimer scopeTimer(m_verbosity, "Total Minibatch read time = %.8g\n");
        bool skip = false;
        Minibatch mb;
        do
        {
            m_mbNumTimeSteps = m_numFramesToProcess[0];
            if (m_noData && m_mbNumTimeSteps == 0)    //no data left for the first channel of this minibatch, 
            {
                mb.atEndOfEpoch = true;
                return mb;
            }

            // skip = (!m_partialMinibatch && (m_mbiter->requestedframes() != m_mbNumTimeSteps) && (m_frameSource->totalframes() > m_mbNumTimeSteps));
            // false. Not clear why we would have this condition for the frame mode. 
            // if (skip)
            // {
                //ReNewBufferForMultiIO(0);
                // }
        }
        while (skip); // keep going if we didn't get the right size minibatch

        m_pMBLayout->Init(m_mbNumTimeSteps, 1, false/*ignored*/);
        if (m_mbNumTimeSteps > 0)
        {
            FillOneUttDataforParallelmode(0, m_mbNumTimeSteps, 0, 0);
        }

        ReNewBufferForMultiIO(0);
        PackToMinibatch(mb);

        mb.atEndOfEpoch = false;
        return mb;
    }

    void FrameModePacker::PackToMinibatch(Minibatch &mb)
    {
        // Filling in the minibatch.
        for (auto name : m_nameToTypeMap)
        {
            if (m_nameToTypeMap[name.first] == InputOutputTypes::real)
            {
                size_t id = m_featureNameToIdMap[name.first];
                size_t dim = m_featureNameToDimMap[name.first];

                auto layout = std::make_shared<Layout>();
                layout->columns = m_pMBLayout;

                std::vector<size_t> dimensions;
                dimensions.push_back(dim);
                layout->rows = std::make_shared<ImageLayout>(dimensions);

                mb.minibatch[m_nameToId[name.first]] =
                    std::make_shared<Input>(m_featuresBufferMultiIO[id].get(), dim * m_mbNumTimeSteps * m_numSeqsPerMB * m_elementSize, layout);
            }
            else if (m_nameToTypeMap[name.first] == InputOutputTypes::category)
            {
                size_t id = m_labelNameToIdMap[name.first];
                size_t dim = m_labelNameToDimMap[name.first];

                auto layout = std::make_shared<Layout>();
                layout->columns = m_pMBLayout;


                std::vector<size_t> dimensions;
                dimensions.push_back(dim);
                layout->rows = std::make_shared<ImageLayout>(dimensions);

                mb.minibatch[m_nameToId[name.first]] =
                    std::make_shared<Input>(m_labelsBufferMultiIO[id].get(), dim * m_mbNumTimeSteps * m_numSeqsPerMB * m_elementSize, layout);
            }
        }
    }

    // copy an utterance into the minibatch given a location (parallel-sequence index, start frame)
    // TODO: This should use DataSlice(). But for that, DataSlice() will have to move out from ComputationNode.
    void FrameModePacker::FillOneUttDataforParallelmode(
        size_t startFr,
        size_t framenum,
        size_t channelIndex,
        size_t parallelSequenceNumber)
    {
        size_t id;
        size_t dim;
        size_t numOfFea = m_featuresBufferMultiIO.size();

        for (auto name: m_nameToId)
        {
            if (m_nameToTypeMap[name.first] == InputOutputTypes::real)
            {
                id = m_featureNameToIdMap[name.first];
                dim = m_featureNameToDimMap[name.first];

                if (m_featuresBufferMultiIO[id] == nullptr || m_featuresBufferAllocatedMultiIO[id] < dim*m_mbNumTimeSteps*m_numSeqsPerMB)
                {
                    m_featuresBufferMultiIO[id] = AllocateExternalBuffer(dim*m_mbNumTimeSteps*m_numSeqsPerMB, m_elementSize);
                    memset(m_featuresBufferMultiIO[id].get(), 0, m_elementSize*dim*m_mbNumTimeSteps*m_numSeqsPerMB);
                    m_featuresBufferAllocatedMultiIO[id] = dim*m_mbNumTimeSteps*m_numSeqsPerMB;
                }

                for (size_t j = 0, k = startFr; j < framenum; j++, k++) // column major, so iterate columns
                {
                    // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                    memcpy_s(
                        &((char*)m_featuresBufferMultiIO[id].get())[(k*m_numSeqsPerMB + channelIndex)*dim*m_elementSize],
                        m_elementSize*dim, 
                        &((char*)m_featuresBufferMultiUtt[parallelSequenceNumber].get())[(j*dim + m_featuresStartIndexMultiUtt[id + parallelSequenceNumber*numOfFea])*m_elementSize],
                        m_elementSize*dim);
                }
            }
            else if (m_nameToTypeMap[name.first] == InputOutputTypes::category)
            {
                id = m_labelNameToIdMap[name.first];
                dim = m_labelNameToDimMap[name.first];
                if (m_labelsBufferMultiIO[id] == nullptr || m_labelsBufferAllocatedMultiIO[id] < dim*m_mbNumTimeSteps*m_numSeqsPerMB)
                {
                    m_labelsBufferMultiIO[id] = AllocateExternalBuffer(dim*m_mbNumTimeSteps*m_numSeqsPerMB, m_elementSize);
                    memset(m_labelsBufferMultiIO[id].get(), 0, m_elementSize*dim*m_mbNumTimeSteps*m_numSeqsPerMB);
                    m_labelsBufferAllocatedMultiIO[id] = dim*m_mbNumTimeSteps*m_numSeqsPerMB;
                }

                for (size_t j = 0, k = startFr; j < framenum; j++, k++)
                {
                    memcpy_s(
                            &((char*)m_labelsBufferMultiIO[id].get())[(k*m_numSeqsPerMB + channelIndex)*dim*m_elementSize],
                            m_elementSize*dim,
                            &((char*)m_labelsBufferMultiUtt[parallelSequenceNumber].get())[(j*dim + m_labelsStartIndexMultiUtt[id + parallelSequenceNumber*numOfFea])*m_elementSize],
                            m_elementSize*dim);
                }
            }
        }
    }

    void FrameModePacker::ReNewBufferForMultiIO(size_t parallelSequenceNumber)
    {
        if (m_noData)
        {
            if (parallelSequenceNumber == 0)
                m_numFramesToProcess[parallelSequenceNumber] = 0;
            return;
        }

        std::vector<std::map<size_t, Sequence>> sequences;
        for (size_t currentIndex = 0; currentIndex < m_requestedMBSize; ++currentIndex)
        {
            auto sequence = m_transformer->GetNextSequence();
            if (sequence.m_endOfEpoch)
            {
                m_noData = true;
                break;
            }

            if (!sequence.m_data.empty())
            {
                sequences.push_back(sequence.m_data);
            }
        }

        if (sequences.size() == 0)
        {
            m_numFramesToProcess[parallelSequenceNumber] = 0;
            return;
        }

        const auto& inputs = m_transformer->GetInputs();
        size_t numOfFea = m_featuresBufferMultiIO.size();
        size_t numOfLabel = m_labelsBufferMultiIO.size();
        size_t totalFeatNum = 0;
        size_t featureSequenceIndex = parallelSequenceNumber*numOfFea;
        for (auto it = m_featureNameToIdMap.begin(); it != m_featureNameToIdMap.end(); ++it)
        {
            size_t id = m_featureNameToIdMap[it->first];
            size_t inputId = m_nameToId[it->first];

            size_t dim = inputs[inputId]->sampleLayout->GetNumElements();
            const size_t actualmbsizeOri = sequences.size();

            m_featuresStartIndexMultiUtt[id + featureSequenceIndex] = totalFeatNum;
            totalFeatNum = dim * actualmbsizeOri + m_featuresStartIndexMultiUtt[id + featureSequenceIndex];
        }

        if ((m_featuresBufferMultiUtt[parallelSequenceNumber] == NULL) || (m_featuresBufferAllocatedMultiUtt[parallelSequenceNumber] < totalFeatNum))
        {
            // eldak : should use simple new
            m_featuresBufferMultiUtt[parallelSequenceNumber] = //AllocateIntermediateBuffer(totalFeatNum, m_elementSize);
                std::shared_ptr<void>(new char[totalFeatNum * m_elementSize], [](char* p) { delete[] p; });
            m_featuresBufferAllocatedMultiUtt[parallelSequenceNumber] = totalFeatNum;
        }

        size_t totalLabelsNum = 0;
        size_t labelSequenceIndex = parallelSequenceNumber*numOfLabel;
        for (auto it = m_labelNameToIdMap.begin(); it != m_labelNameToIdMap.end(); ++it)
        {
            size_t id = m_labelNameToIdMap[it->first];
            size_t inputId = m_nameToId[it->first];

            size_t dim = inputs[inputId]->sampleLayout->GetNumElements();
            const size_t actualmbsizeOri = sequences.size();

            m_labelsStartIndexMultiUtt[id + labelSequenceIndex] = totalLabelsNum;
            totalLabelsNum = m_labelsStartIndexMultiUtt[id + labelSequenceIndex] + dim * actualmbsizeOri;
        }

        if ((m_labelsBufferMultiUtt[parallelSequenceNumber] == NULL) || (m_labelsBufferAllocatedMultiUtt[parallelSequenceNumber] < totalLabelsNum))
        {
            // eldak: should use simple new.
            m_labelsBufferMultiUtt[parallelSequenceNumber] = //AllocateIntermediateBuffer(totalLabelsNum, m_elementSize);
                std::shared_ptr<void>(new char[totalLabelsNum * m_elementSize], [](char* p) {delete[] p; });

            m_labelsBufferAllocatedMultiUtt[parallelSequenceNumber] = totalLabelsNum;
        }

        memset(m_labelsBufferMultiUtt[parallelSequenceNumber].get(), 0, m_elementSize*totalLabelsNum);

        bool first = true;
        for (auto it = m_featureNameToIdMap.begin(); it != m_featureNameToIdMap.end(); ++it)
        {
            size_t id = m_featureNameToIdMap[it->first];
            size_t inputId = m_nameToId[it->first];

            size_t fdim = inputs[inputId]->sampleLayout->GetNumElements();
            const size_t actualmbsizeOri = sequences.size();

            if (first)
            {
                m_numFramesToProcess[parallelSequenceNumber] = actualmbsizeOri;
                first = false;
            }
            else
            {
                if (m_numFramesToProcess[parallelSequenceNumber] != actualmbsizeOri)
                {
                    RuntimeError("The multi-IO features has inconsistent number of frames!");
                }
            }
            assert(actualmbsizeOri == sequences.size());


            for (int k = 0; k < actualmbsizeOri; k++) // column major, so iterate columns
            {
                const void* sequence = sequences[k][inputId].data;

                // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                memcpy_s(&((char*)m_featuresBufferMultiUtt[parallelSequenceNumber].get())[(k*fdim + m_featuresStartIndexMultiUtt[id + featureSequenceIndex]) * m_elementSize], m_elementSize*fdim, sequence, m_elementSize*fdim);
            }
        }

        for (auto it = m_labelNameToIdMap.begin(); it != m_labelNameToIdMap.end(); ++it)
        {
            size_t id = m_labelNameToIdMap[it->first];
            size_t inputId = m_nameToId[it->first];

            size_t fdim = inputs[inputId]->sampleLayout->GetNumElements();
            const size_t actualmbsizeOri = sequences.size();

            if (first)
            {
                m_numFramesToProcess[parallelSequenceNumber] = actualmbsizeOri;
                first = false;
            }
            else
            {
                if (m_numFramesToProcess[parallelSequenceNumber] != actualmbsizeOri)
                {
                    RuntimeError("The multi-IO features has inconsistent number of frames!");
                }
            }
            assert(actualmbsizeOri == sequences.size());


            for (int k = 0; k < actualmbsizeOri; k++) // column major, so iterate columns
            {
                const void* sequence = sequences[k][inputId].data;

                // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                memcpy_s(&((char*)m_labelsBufferMultiUtt[parallelSequenceNumber].get())[(k*fdim + m_labelsStartIndexMultiUtt[id + labelSequenceIndex]) * m_elementSize], m_elementSize*fdim, sequence, m_elementSize*fdim);
            }
        }

        for (const auto& s : sequences)
        {
            for(const auto& i : s)
            {
                delete[] i.second.data;
            }
        }
    }

    std::shared_ptr<void> FrameModePacker::AllocateExternalBuffer(size_t numElements, size_t elementSize)
    {
        return std::shared_ptr<void>(
            m_memoryProvider->Alloc(elementSize, numElements),
            [this](void* p) { m_memoryProvider->Free(p); });
    }
}}}
