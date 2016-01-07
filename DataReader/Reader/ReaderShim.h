//
// <copyright file="HTKMLFReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReaderShim.cpp: header for shim that wraps new HTKMLF reader
//

#pragma once

#include <map>
#include <string>
#include "DataReader.h"
#include "commandArgUtil.h"
#include "SubstitutingMemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    typedef ReaderPtr(*ReaderFactory)(const ConfigParameters& parameters);

    template<class ElemType>
    class ReaderShim : public IDataReader<ElemType>
    {
        ReaderPtr m_reader;
        ReaderFactory m_factory;
        bool m_endOfEpoch;

        MBLayoutPtr m_layout;

        std::map<std::wstring, size_t> m_nameToInputId;
        std::vector<InputDescriptionPtr> m_inputs;

    public:
        explicit ReaderShim(ReaderFactory factory);
        virtual ~ReaderShim() {}

        virtual void Init(const ScriptableObjects::IConfigRecord & /*config*/) override { assert(false); }
        virtual void Init(const ConfigParameters & config) override;

        virtual void Destroy() override { delete this; }

        virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples) override;
        virtual void StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples) override;

        virtual bool SupportsDistributedMBRead() const override
        {
            return true;
        }

        virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices) override;

        virtual bool DataEnd(EndDataType endDataType) override;

        void CopyMBLayoutTo(MBLayoutPtr) override;

        virtual size_t GetNumParallelSequences() override;
    };
}}}
