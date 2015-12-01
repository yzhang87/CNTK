//
// <copyright file="LibSVMBinaryReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// LibSVMBinaryReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
#include "stdafx.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "commandArgUtil.h"
#include "RandomOrdering.h"
#include <string>
#include <map>
#include <vector>
#include <random>

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

            enum LabelKind
            {
                labelNone = 0,  // no labels to worry about
                labelCategory = 1, // category labels, creates mapping tables
                labelRegression = 2,  // regression labels
                labelOther = 3, // some other type of label
            };

            template<class ElemType>
            class SparseBinaryInput {
            private:
#ifdef _WIN32
                HANDLE m_hndl;
                HANDLE m_filemap;
#else
                int m_hndl;
                int64_t m_fileSize;
#endif
                int64_t header_size;
				
                DWORD sysGran;

                //void* header_orig; // Don't need this since the header is at the start of the file
                void* offsets_orig;
                void* data_orig;

                void* header_buffer;
                int64_t* offsets_buffer;
                void* data_buffer;

                std::vector<std::wstring> m_features;
                std::vector<std::wstring> m_labels;

                int64_t numRows;
                int64_t numBatches;
                std::map<std::wstring, int32_t> mappedNumCols;

                size_t minibatchSize;

                int32_t numFeatures;
                int32_t numLabels;
                
                size_t m_startMB;
                size_t m_endMB;
                size_t m_windowSize;

                size_t m_lower;

                int64_t m_dataOffset;
                int64_t m_dataPadding;
                
                ElemType* DSSMLabels;
                size_t DSSMCols;
                int64_t m_windowSizeBytes;

            public:
                SparseBinaryInput() {};
                ~SparseBinaryInput();

                //void Init(std::wstring fileName, std::vector<std::wstring> features, std::vector<std::wstring> labels);
                void Init(std::wstring fileName, std::map<std::wstring, std::wstring> rename);
                void StartMinibatchLoop(size_t startMB, size_t endMB, size_t windowSize);
                size_t Next_Batch(std::map<std::wstring, Matrix<ElemType>*>& matrices, size_t batchIndex);
				size_t Load_Window(size_t lowerBound);
				void Unload_Window();
                void Dispose();

                size_t getMBSize() { return minibatchSize; };
                int64_t getNumMB() { return numBatches; };
            };

            template<class ElemType>
            class LibSVMBinaryReader : public IDataReader<ElemType>
            {
                public:
                   // typedef unsigned LabelIdType;
                    //typedef std::string LabelType;
                    using LabelType = typename IDataReader<ElemType>::LabelType;
                    using LabelIdType = typename IDataReader<ElemType>::LabelIdType;
            public:
				virtual void Init(const ConfigParameters & config) override { InitFromConfig(config); }
				virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }
				template<class ConfigRecordType> void InitFromConfig(const ConfigRecordType &);
                virtual void Destroy();

                LibSVMBinaryReader() { read_order = nullptr; m_pMBLayout=make_shared<MBLayout>(); };
                virtual ~LibSVMBinaryReader();

                virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
				virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples = requestDataSize) override;
                virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    
                virtual bool SupportsDistributedMBRead() const override { return true; }


				template<class ConfigRecordType> void RenamedMatrices(const ConfigRecordType& readerConfig, std::map<std::wstring, std::wstring>& rename);
                virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
                //virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
                //virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, typename LabelType>& labelMapping);
                virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
                virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);
        
                size_t GetNumParallelSequences() { return m_pMBLayout->GetNumParallelSequences(); }
				void CopyMBLayoutTo(MBLayoutPtr pMBLayout) { pMBLayout->CopyFrom(m_pMBLayout); };

                virtual bool DataEnd(EndDataType endDataType);

                size_t NumberSlicesInEachRecurrentIter() { return 1; }
                void SetNbrSlicesEachRecurrentIter(const size_t) { };
                void SetSentenceEndInBatch(std::vector<size_t> &/*sentenceEnd*/){};

            private:
                bool Randomize();
                void FillReadOrder(size_t lowerBound, size_t windowSize);
                void Shuffle( size_t lowerBound, size_t windowSize);
                void ReleaseMemory();

				MBLayoutPtr m_pMBLayout;
                ConfigParameters m_readerConfig;

                SparseBinaryInput<ElemType> dataInput;

                std::map<std::wstring, std::wstring> m_rename;
                std::vector<std::wstring> m_features;
                std::vector<std::wstring> m_labels;

                size_t* read_order; // array to shuffle to reorder the dataset
                size_t read_order_length;

                unsigned long m_randomize; // randomization range

                size_t m_overflowOffset;
                size_t m_mbSize;    // size of minibatch requested
                size_t m_numBatches;    // size of minibatch requested
                size_t m_numWindows;
                size_t m_nextMB; // starting sample # of the next minibatch
                size_t m_readMB; // starting sample # of the next minibatch

                size_t m_requestedEpochSize; // size of an epoch
                size_t m_epochSize; // size of an epoch
                size_t m_epoch; // which epoch are we on

                size_t m_startMB;
                size_t m_endMB;
                size_t m_curLower;
                
                size_t m_subsetNum;
                size_t m_numSubsets;

                size_t m_windowSize;
                size_t m_curWindowSize;

                bool m_partialMinibatch;    // a partial minibatch is allowed
                size_t m_traceLevel;

				RandomOrdering m_randomordering;   // randomizing class

                std::mt19937_64 random_engine;

                std::map<LabelIdType, LabelType> m_mapIdToLabel;
                std::map<LabelType, LabelIdType> m_mapLabelToId;
            };
        }
    }
}
