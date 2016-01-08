//
// <copyright file="HTKMLFReaderShim.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReaderShim.cpp: implementation for shim that wraps new HTKMLF reader
//

#include "stdafx.h"
#ifdef _WIN32
#include <objbase.h>
#endif
#include "Basics.h"

#include "htkfeatio.h"                  // for reading HTK features
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)

#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "commandArgUtil.h"
#include "HTKMLFReaderShim.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

#ifdef __unix__
// TODO probably not needed anymore
#include <limits.h>
#endif

#include "SubstitutingMemoryProvider.h"
#include "CudaMemoryProvider.h"
#include "HeapMemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    void HTKMLFReaderShim<ElemType>::Init(const ConfigParameters& config)
    {
        m_layout = make_shared<MBLayout>();

        assert(config(L"frameMode", true));
        m_memoryProvider = std::make_shared<HeapMemoryProvider>();
        m_packer = std::make_shared<FrameModePacker>(config, m_memoryProvider, sizeof(ElemType));

        intargvector numberOfuttsPerMinibatchForAllEpochs =
            config(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int>{ 1 })));

        auto numSeqsPerMBForAllEpochs = numberOfuttsPerMinibatchForAllEpochs;
        m_layout->Init(numSeqsPerMBForAllEpochs[0], 0, true);
    }

    template<class ElemType>
    void HTKMLFReaderShim<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize)
    {
        return StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, requestedEpochSamples);
    }

    template<class ElemType>
    void HTKMLFReaderShim<ElemType>::StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /*= requestDataSize*/)
    {
        EpochConfiguration config;
        config.workerRank = subsetNum;
        config.numberOfWorkers = numSubsets;
        config.minibatchSize = requestedMBSize;
        config.totalSize = requestedEpochSamples;
        config.index = epoch;

        m_packer->StartEpoch(config);
    }

    template<class ElemType>
    bool HTKMLFReaderShim<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
    {
        // eldak: Hack.
        int deviceId = matrices.begin()->second->GetDeviceId();
        for (auto mx : matrices)
        {
            if (mx.second->GetDeviceId() != deviceId)
            {
                assert(false);
            }
        }

        Minibatch m = m_packer->ReadMinibatch();
        if (m.atEndOfEpoch)
        {
            return false;
        }

        auto inputs = m_packer->GetInputs();
        std::map<size_t, wstring> idToName;
        for (auto i: inputs)
        {
            idToName.insert(std::make_pair(i->id, i->name));
        }

        for (int i = 0; i < m.minibatch.size(); i++)
        {
            const auto& input = m.minibatch[i];
            const std::wstring& name = idToName[i];
            if (matrices.find(name) == matrices.end())
            {
                continue;
            }

            auto layout = input->layout;
            size_t columnNumber = layout->columns->GetNumCols();
            size_t rowNumber = layout->rows->GetNumElements();

            // Current hack.
            m_layout = layout->columns;

            auto data = reinterpret_cast<const ElemType*>(input->data);
            matrices[name]->SetValue(rowNumber, columnNumber, matrices[name]->GetDeviceId(), const_cast<ElemType*>(data), matrixFlagNormal);
        }

        return m;
    }

    template<class ElemType>
    bool HTKMLFReaderShim<ElemType>::DataEnd(EndDataType /*endDataType*/)
    {
        return false;
    }

    template<class ElemType>
    void HTKMLFReaderShim<ElemType>::CopyMBLayoutTo(MBLayoutPtr layout)
    {
        layout->CopyFrom(m_layout);
    }

    template<class ElemType>
    size_t HTKMLFReaderShim<ElemType>::GetNumParallelSequences()
    {
        return m_layout->GetNumParallelSequences();  // (this function is only used for validation anyway)
    }

    template class HTKMLFReaderShim<float>;
    template class HTKMLFReaderShim<double>;
}}}
