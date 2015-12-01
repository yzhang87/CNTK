//
// <copyright file="LibSVMBinaryReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// LibSVMBinaryReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "LibSVMBinaryReader.h"
#include "fileutil.h"   // for fexists()
#include <random>
#include <map>
#include <ctime>
#ifndef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#endif

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

            DWORD HIDWORD(size_t size) { return size >> 32; }
            DWORD LODWORD(size_t size) { return size & 0xFFFFFFFF; }

            template<class ElemType>
            SparseBinaryInput<ElemType>::~SparseBinaryInput(){
                Dispose();
            }

			template<class ElemType>
            SparseBinaryFile<ElemType>::~SparseBinaryFile() {
                Dispose();
            }

			template<class ElemType>
            void SparseBinaryFile<ElemType>::Init(wstring fileName) {
#ifdef _WIN32
                m_hndl = CreateFile(fileName.c_str(), GENERIC_READ,
                    FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
                if (m_hndl == INVALID_HANDLE_VALUE)
                {
                    RuntimeError("Unable to Open/Create file %ls, error %x", fileName.c_str(), GetLastError());
                    /*
                    char message[1024];
                    sprintf_s(message, 1024, "Unable to Open/Create file %ls, error %x", fileName.c_str(), GetLastError());
                    throw runtime_error(message);
                    */
                }

                m_filemap = CreateFileMapping(m_hndl, NULL, PAGE_READONLY, 0, 0, NULL);

                SYSTEM_INFO sysinfo;
                GetSystemInfo(&sysinfo);
                sysGran = sysinfo.dwAllocationGranularity;

                m_headerSize = sizeof(int64_t) * 2000000;
                m_headerBuffer = MapViewOfFile(m_filemap,   // handle to map object
                    FILE_MAP_READ, // get correct permissions
                    HIDWORD(0),
                    LODWORD(0),
                    m_headerSize);
#else
                sysGran = sysconf(_SC_PAGESIZE);
                m_hndl = open(msra::strfun::utf8(fileName).c_str(), O_RDONLY, O_CREAT);
                if (m_hndl < 0)
                {
                    RuntimeError("Unable to Open/Create file %s, error %d", msra::strfun::utf8(fileName).c_str(), errno);
                }
                m_headerBuffer = mmap(0, m_headerSize, PROT_READ, MAP_SHARED, m_hndl, 0);
#endif

            }

			template<class ElemType>
            void SparseBinaryFile<ElemType>::SetOffsets(int64_t base_offset, int64_t numBatches) {

                m_numBatches = numBatches;

                int64_t offsets_padding = base_offset % sysGran;
                base_offset -= offsets_padding;

                m_headerSize = numBatches*sizeof(int64_t) + offsets_padding;

#ifdef _WIN32
                m_offsetsOrig = MapViewOfFile(m_filemap,   // handle to map object
                    FILE_MAP_READ, // get correct permissions
                    HIDWORD(base_offset),
                    LODWORD(base_offset),
                    m_headerSize);
#else
                m_offsetsOrig = mmap(0, m_headerSize, PROT_READ, MAP_SHARED, m_hndl, base_offset);
#endif

                m_offsetsBuffer = (int64_t*)((char*)m_offsetsOrig + offsets_padding);

                m_dataStart = base_offset + offsets_padding + numBatches * sizeof(int64_t);

                m_dataOrig = NULL;

#ifdef _WIN32
#else
				struct stat stat_buf;
				fstat( m_hndl, &stat_buf );
				m_fileSize = stat_buf.st_size;
#endif

            }

            template<class ElemType>
            //void SparseBinaryInput<ElemType>::Init(std::wstring fileName, std::vector<std::wstring> features, std::vector<std::wstring> labels)
            void SparseBinaryInput<ElemType>::Init(std::wstring fileName, std::map<std::wstring, std::wstring> rename )
            {
                if (m_file != nullptr) {
                    m_file.reset();
                }
                m_file = make_shared<SparseBinaryFile<ElemType>>();
                m_file->Init(fileName);

                void* header_buffer = m_file->GetHeader();
                int64_t base_offset = 0;

                numRows = *(int64_t*)((char*)header_buffer + base_offset);
                base_offset += sizeof(int64_t);

                numBatches = *(int64_t*)((char*)header_buffer + base_offset);
                base_offset += sizeof(int64_t);

                numFeatures = *(int32_t*)((char*)header_buffer + base_offset);
                base_offset += sizeof(int32_t);

                numLabels = *(int32_t*)((char*)header_buffer + base_offset);
                base_offset += sizeof(int32_t);

                int32_t len;
                int32_t numCols;
                for (int32_t c = 0; c < numFeatures; c++)
                {
                    len = *(int32_t*)((char*)header_buffer + base_offset);
                    base_offset += sizeof(int32_t);

                    std::string name((char*)header_buffer + base_offset, len);
                    std::wstring wname = msra::strfun::utf16(name);
                    if (rename.find(wname) == rename.end())
                    {
                        m_features.emplace_back(wname);
                    }
                    else
                    {
                        m_features.emplace_back(rename[wname]);
                    }
                    base_offset += sizeof(int8_t)*len;

                    numCols = *(int32_t*)((char*)header_buffer + base_offset);
                    //numCols = (int32_t)49292;
                    base_offset += sizeof(int32_t);
                    mappedNumCols[m_features.back()] = numCols;

                }
                for (int32_t c = 0; c < numLabels; c++)
                {
                    len = *(int32_t*)((char*)header_buffer + base_offset);
                    base_offset += sizeof(int32_t);

                    std::string name((char*)header_buffer + base_offset, len);
                    std::wstring wname = msra::strfun::utf16(name);
                    if (rename.find(wname) == rename.end())
                    {
                        m_labels.emplace_back(wname);
                    }
                    else
                    {
                        //m_features.emplace_back(rename[wname]);
                        m_labels.emplace_back(rename[wname]);
                    }
                    base_offset += sizeof(int8_t)*len;

                    numCols = *(int32_t*)((char*)header_buffer + base_offset);
                    base_offset += sizeof(int32_t);
                    mappedNumCols[m_labels.back()] = numCols;

                }

                m_file->SetOffsets(base_offset, numBatches);
                m_file->ReleaseHeader();

            }
            
            template<class ElemType>
            void SparseBinaryFile<ElemType>::ReleaseHeader() {

                if (m_headerBuffer != NULL) {
#ifdef _WIN32
                    UnmapViewOfFile(m_headerBuffer);
#else
                    munmap(m_headerBuffer,m_headerSize);
                    m_headerSize = 0;
#endif
                }
                m_headerBuffer = NULL;
            }

            
            template<class ElemType>
            void SparseBinaryFile<ElemType>::Unload_Window() {

                if (m_dataOrig != NULL) {
#ifdef _WIN32
                    UnmapViewOfFile(m_dataOrig);
#else
                    munmap(m_dataOrig,m_dataSize);
                    m_dataSize = 0;
#endif
                }
				m_dataOrig = NULL;
                m_dataBuffer = NULL;
            }

            template<class ElemType>
            void SparseBinaryInput<ElemType>::Unload_Window() {
                m_file->Unload_Window();
            }

            template<class ElemType>
            void SparseBinaryFile<ElemType>::StartMinibatchLoop(size_t startMB, size_t endMB, size_t windowSize) {
                m_startMB = startMB;
                m_endMB = endMB;
                m_maxWindowSize = windowSize;
            }

            template<class ElemType>
            void SparseBinaryInput<ElemType>::StartMinibatchLoop(size_t startMB, size_t endMB, size_t windowSize) {
                m_startMB = startMB;
                m_endMB = endMB;
                m_windowSize = windowSize;
                m_file->StartMinibatchLoop(startMB, endMB, windowSize);
            }

            template<class ElemType>
            size_t SparseBinaryInput<ElemType>::Load_Window(size_t lowerBound) {
                return m_file->Load_Window(lowerBound);
            }

            template<class ElemType>
            size_t SparseBinaryFile<ElemType>::Load_Window(size_t lowerBound ) {
                Unload_Window();
                m_mappedLower = lowerBound;
				size_t upper = lowerBound + m_maxWindowSize;
				if (upper > (size_t)m_numBatches) {
                    upper = m_numBatches;
#ifdef _WIN32
					m_dataSize = 0;
#else
					m_dataSize = m_fileSize - offsets_buffer[m_mappedLower];
#endif
				}
                else {
					if (upper > m_endMB) {
                        upper = m_endMB;
                        m_curWindowSize = upper - m_mappedLower;
                    }
                    m_dataSize = m_offsetsBuffer[upper] - m_offsetsBuffer[m_mappedLower];
                }
				size_t m_curWindowSize = upper - m_mappedLower;
                //m_dataOffset = header_offset;
                int64_t dataOffset = m_dataStart + m_offsetsBuffer[m_mappedLower];
                int64_t dataPadding = dataOffset % sysGran;

                dataOffset -= dataPadding;
                m_dataSize += dataPadding;

                fprintf(stderr, "loading window: [%ld - %ld]\n", m_mappedLower, upper);
#ifdef _WIN32
                m_dataOrig = MapViewOfFile(m_filemap,   // handle to map object
                    FILE_MAP_READ, // get correct permissions
                    HIDWORD(dataOffset),
                    LODWORD(dataOffset),
                    m_dataSize );
#else

                m_dataOrig = mmap(0, m_dataSize, PROT_READ, MAP_SHARED, m_hndl, dataOffset);
#endif
                m_dataBuffer = (char*)m_dataOrig + dataPadding;

                return m_curWindowSize;
            }
            
            template<class ElemType>
            void* SparseBinaryFile<ElemType>::GetMinibatch(size_t cur_batch){
                int64_t buffer_offset = m_offsetsBuffer[m_mappedLower + cur_batch] - m_offsetsBuffer[m_mappedLower];
                void* data_buffer = (char*)m_dataBuffer + buffer_offset;

                return data_buffer;
            }

            template<class ElemType>
            size_t SparseBinaryInput<ElemType>::Next_Batch(std::map<std::wstring, Matrix<ElemType>*>& matrices, size_t cur_batch){

                void* data_buffer = m_file->GetMinibatch(cur_batch);
                int64_t buffer_offset = 0;
                int32_t nnz;
                int32_t curMBSize;

                curMBSize = *(int32_t*)((char*)data_buffer + buffer_offset);
                buffer_offset += sizeof(int32_t);

                for (int32_t c = 0; c < m_features.size(); c++)
                {
                    nnz = *(int32_t*)((char*)data_buffer + buffer_offset);
                    buffer_offset += sizeof(int32_t);

                    ElemType* values = (ElemType*)((char*)data_buffer + buffer_offset);
                    buffer_offset += sizeof(ElemType)*nnz;

                    /**/
                    int32_t* rowIndices = (int32_t*)((char*)data_buffer + buffer_offset);
                    buffer_offset += sizeof(int32_t)*nnz;
                    /**/
                    /*
                    int32_t* rowIndices = (int32_t*)malloc(sizeof(int32_t)*nnz);
                    memcpy(rowIndices, (char*)data_buffer + buffer_offset, nnz*sizeof(int32_t));
                    for (int32_t d = 0; d < nnz; d++)
                    {
                    if (rowIndices[d] > 49291)
                    {
                    fprintf(stderr, "fixing index %d = %d\n", d, rowIndices[d]);
                    rowIndices[d] = 1;
                    }
                    }
                    buffer_offset += sizeof(int32_t)*nnz;
                    */

                    int32_t* colIndices = (int32_t*)((char*)data_buffer + buffer_offset);
                    buffer_offset += sizeof(int32_t)*(curMBSize + 1);

                    auto findMat = matrices.find(m_features[c]);
                    if (findMat != matrices.end())
                    {
                        auto mat = findMat->second;
                        mat->SetMatrixFromCSCFormat(colIndices, rowIndices, values, nnz, mappedNumCols[m_features[c]], curMBSize);
#ifdef DEBUG
                        mat->Print("features");
#endif
                    }
                    //free(rowIndices);
                }

                for (int32_t c = 0; c < m_labels.size(); c++)
                {
                    int32_t numCols = mappedNumCols[m_labels[c]];

                    ElemType* m_labelsBuffer = (ElemType*)((char*)data_buffer + buffer_offset);
                    buffer_offset += sizeof(ElemType)*(curMBSize* numCols);

                    auto findMat = matrices.find(m_labels[c]);
                    if (findMat != matrices.end())
                    {
                        auto mat = findMat->second;
                        mat->SetValue(numCols, curMBSize, mat->GetDeviceId(), m_labelsBuffer, matrixFlagNormal);
#ifdef DEBUG
                        mat->Print("labels");
#endif
                    }
                }
                auto findMat = matrices.find(L"DSSMLabel");
                if( findMat != matrices.end())
                {
                    auto mat = findMat->second;
                    size_t numRows = mat->GetNumRows();
                    if (DSSMCols < curMBSize) {
                        if (DSSMLabels != nullptr) {
                            free(DSSMLabels);
                        }
                        DSSMCols = curMBSize;
                        DSSMLabels = (ElemType*)malloc(sizeof(ElemType)*numRows*curMBSize);
                        memset(DSSMLabels, 0, sizeof(ElemType)*numRows*curMBSize);
                        for (size_t c = 0; c < curMBSize; c += numRows) {
                            DSSMLabels[c] = 1;
                        }
                    }
                    if (mat->GetNumCols() != curMBSize) {
                        mat->SetValue(mat->GetNumRows(), curMBSize, mat->GetDeviceId(), DSSMLabels, matrixFlagNormal);
                    }

                }
                return (size_t)curMBSize;
            }

            template<class ElemType>
            void SparseBinaryFile<ElemType>::Dispose(){
                if (m_offsetsOrig != NULL){
#ifdef _WIN32
                    UnmapViewOfFile(m_offsetsOrig);
#else
                    munmap(m_offsetsOrig,m_headerSize);
#endif

                }
                if (m_dataOrig != NULL)
                {
#ifdef _WIN32
                    UnmapViewOfFile(m_dataOrig);
#else
                    munmap(m_dataOrig,m_dataSize);
#endif
                }
            }

            template<class ElemType>
            void SparseBinaryInput<ElemType>::Dispose(){
                if (m_file != nullptr) {
                    m_file->Dispose();
                }
                m_file.reset();
                if (DSSMLabels != nullptr) {
                    delete[] DSSMLabels;
                }

            }


            // Init - Reader Initialize for multiple data sets
            // config - [in] configuration parameters for the datareader
            // Sample format below:
            //# Parameter values for the reader
            //reader=[
            //  # reader to use
            //  readerType=LibSVMBinaryReader
            //  miniBatchMode=Partial
            //  randomize=None
            //  features=[
            //    dim=784
            //    start=1
            //    file=c:\speech\mnist\mnist_test.txt
            //  ]
            //  labels=[
            //    dim=1
            //      start=0
            //      file=c:\speech\mnist\mnist_test.txt
            //      labelMappingFile=c:\speech\mnist\labels.txt
            //      labelDim=10
            //      labelType=Category
            //  ]
            //]

            template<class ElemType>
			template<class ConfigRecordType>

            void LibSVMBinaryReader<ElemType>::RenamedMatrices(const ConfigRecordType& config, std::map<std::wstring, std::wstring>& rename)
            {
				for (const auto & id : config.GetMemberIds())
				{
					if (!config.CanBeConfigRecord(id))
						continue;
					const ConfigRecordType & temp = config(id);
					// see if we have a config parameters that contains a "dim" element, it's a sub key, use it
                    if (temp.ExistsCurrent(L"rename"))
                    {

                        std::wstring ren = temp(L"rename");
                        rename.emplace(msra::strfun::utf16(id), msra::strfun::utf16(ren));
                    }
                }
            }


            template<class ElemType>
			template<class ConfigRecordType>
			void LibSVMBinaryReader<ElemType>::InitFromConfig(const ConfigRecordType & readerConfig)
            {
                // Determine the names of the features and lables sections in the config file.
                // features - [in,out] a vector of feature name strings
                // labels - [in,out] a vector of label name strings
                // For SparseBinary dataset, we only need features. No label is necessary. The following "labels" just serves as a place holder
                GetFileConfigNames(readerConfig, m_features, m_labels);
                RenamedMatrices(readerConfig, m_rename);

                m_epoch = 0;

                m_partialMinibatch = false;
                m_traceLevel = (size_t)readerConfig(L"traceLevel", 0);

                if (readerConfig.Exists(L"randomize"))
                {
                    string randomizeString = readerConfig(L"randomize");
                    if (randomizeString == "None")
                    {
                        m_randomize = 0L;
                    }
                    else if (randomizeString == "Auto")
                    {
                        //m_randomize = randomizeAuto;
                        time_t rawtime;
                        struct tm* timeinfo;
                        time(&rawtime);
                        timeinfo = localtime(&rawtime);
                        m_randomize = (unsigned long)(timeinfo->tm_sec + timeinfo->tm_min * 60 + timeinfo->tm_hour * 60 * 60);
                    }
                    else
                    {
                        m_randomize = readerConfig(L"randomize", 0 );
                        //m_randomize = 0L;
                    }
                }
                else
                {
                    m_randomize = 0L;
                }


                std::string minibatchMode(readerConfig(L"minibatchMode", "Partial"));
                m_partialMinibatch = !_stricmp(minibatchMode.c_str(), "Partial");

                std::wstring file = readerConfig(L"file", L"");

                m_windowSize = readerConfig(L"windowSize", 10000);

                dataInput.Init(file, m_rename);

                m_mbSize = (size_t)readerConfig(L"minibatch", 0);
                if (m_mbSize > 0)
                {
                    if (dataInput.getMBSize() != m_mbSize)
                    {
                        RuntimeError("Data file and config file have mismatched minibatch sizes.\n");
                        return;
                    }
                }
                else
                {
                    m_mbSize = dataInput.getMBSize();
                }

                m_numBatches = dataInput.getNumMB();

                m_epochSize = readerConfig(L"epochMinibatches", 0);
                if (m_epochSize > 0)
                {
                    if (m_numBatches < m_epochSize)
                    {
                        if (m_traceLevel > 0)
                        {
                            fprintf(stderr, "Warning: epoch size is larger than input data size. Data will be reused.\n");
                        }
                    }
                }
                else {
                    m_epochSize = (size_t)dataInput.getNumMB();
                }

            }

            template<class ElemType>
            void LibSVMBinaryReader<ElemType>::FillReadOrder( size_t /*lowerBound*/, size_t windowSize)
            {
                if (read_order != nullptr)
                {
                    delete[] read_order;
                }
				read_order = new size_t[windowSize];
				for (size_t c = 0; c < windowSize; c++)
				{
					read_order[c] = c;
				}
            }

            // Destroy - cleanup and remove this class
            // NOTE: this destroys the object, and it can't be used past this point
            template<class ElemType>
            void LibSVMBinaryReader<ElemType>::Destroy()
            {
                delete this;
            }

            // destructor - virtual so it gets called properly 
            template<class ElemType>
            LibSVMBinaryReader<ElemType>::~LibSVMBinaryReader()
            {
                ReleaseMemory();
            }
            
            //StartMinibatchLoop - Startup a minibatch loop 
            // mbSize - [in] size of the minibatch (number of Samples, etc.)
            // epoch - [in] epoch number for this loop, if > 0 the requestedEpochSamples must be specified (unless epoch zero was completed this run)
            // requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
            //   this value must be a multiple of mbSize, if it is not, it will be rounded up to one.
            template<class ElemType>
            void LibSVMBinaryReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
            {
                StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, requestedEpochSamples);
            }

            //StartMinibatchLoop - Startup a minibatch loop 
            // mbSize - [in] size of the minibatch (number of Samples, etc.)
            // epoch - [in] epoch number for this loop, if > 0 the requestedEpochSamples must be specified (unless epoch zero was completed this run)
            // requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
            //   this value must be a multiple of mbSize, if it is not, it will be rounded up to one.
            template<class ElemType>
				void LibSVMBinaryReader<ElemType>::StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t /*requestedEpochSamples*/ )
				//void LibSVMBinaryReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t /*requestedEpochSamples*/)
            {

                m_readMB = 0;

                m_epoch = epoch;

                m_mbSize = mbSize;

                m_subsetNum = subsetNum;
                m_numSubsets = numSubsets;
                
                m_epochSize = m_numBatches / numSubsets;

                m_startMB = m_epochSize * subsetNum;
                m_endMB = m_epochSize * ( subsetNum + 1 );

                size_t remainder = m_numBatches % numSubsets;

                size_t lb = min(remainder, subsetNum);
                size_t ub = min(remainder, subsetNum + 1);

                m_epochSize += (subsetNum < remainder ) ? 1 : 0;

                m_windowSize = min(m_windowSize, m_epochSize);
                m_startMB += lb;
                m_endMB += ub;

                m_curLower = m_startMB;

                dataInput.StartMinibatchLoop(m_startMB, m_endMB, m_windowSize);
                m_curWindowSize = dataInput.Load_Window(m_curLower);
                Shuffle(m_curLower, m_curWindowSize);
                /*
                if (mbSize != m_mbSize)
                {
                RuntimeError("Data file and config file have mismatched minibatch sizes.\n");
                }
                */

                //m_epochSize = requestedEpochSamples / m_mbSize;
            }

            // GetMinibatch - Get the next minibatch (features and labels)
            // matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix, 
            //             [out] each matrix resized if necessary containing data. 
            // returns - true if there are more minibatches, false if no more minibatchs remain
            template<class ElemType>
            bool LibSVMBinaryReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
            {
                if (m_readMB >= m_epochSize)
                {
                    return false;
                }

                //fprintf(stderr,"m_nextmb: %ld\treadorder: %ld\n", m_nextMB, read_order[m_nextMB]);
                size_t actualmbsize = dataInput.Next_Batch(matrices, read_order[m_nextMB]);
				m_pMBLayout->Init(actualmbsize, 1, false/*means it is not sequential*/);

                m_readMB++;
                m_nextMB++;

                if (m_nextMB >= m_curWindowSize)
                {
                    m_curLower += m_windowSize;
                    m_curWindowSize = dataInput.Load_Window(m_curLower);
                    m_nextMB = 0;
                }
                /*
                m_readMB++;

                m_nextMB++;
                if (m_nextMB >= m_numBatches)
                {
                    m_nextMB = 0;
                    Shuffle();
                }
				*/

                return true;
            }

            // GetLabelMapping - Gets the label mapping from integer index to label type 
            // returns - a map from numeric datatype to native label type 
            template<class ElemType>
            const std::map<typename IDataReader<ElemType>::LabelIdType, typename IDataReader<ElemType>::LabelType>& LibSVMBinaryReader<ElemType>::GetLabelMapping(const std::wstring&)
            {
                return m_mapIdToLabel;
            }

            // SetLabelMapping - Sets the label mapping from integer index to label 
            // labelMapping - mapping table from label values to IDs (must be 0-n)
            // note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
            template<class ElemType>
            void LibSVMBinaryReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<typename IDataReader<ElemType>::LabelIdType, LabelType>& labelMapping)
            {
                m_mapIdToLabel = labelMapping;
                m_mapLabelToId.clear();
                for (std::pair<unsigned, LabelType> var : labelMapping)
                {
                    m_mapLabelToId[var.second] = var.first;
                }
            }

            template<class ElemType>
            bool LibSVMBinaryReader<ElemType>::DataEnd(EndDataType endDataType)
            {
                bool ret = false;
                switch (endDataType)
                {
                case endDataNull:
                    assert(false);
                    break;
                case endDataEpoch:
                    ret = (m_nextMB >= m_epochSize);
                    break;
                case endDataSet:
                    ret = (m_nextMB >= m_numBatches);
                    break;
                case endDataSentence:  // for fast reader each minibatch is considered a "sentence", so always true
                    ret = true;
                    break;
                }
                return ret;
            }

            // ReleaseMemory - release the memory footprint of LibSVMBinaryReader
            // used when the caching reader is taking over
            template<class ElemType>
            void LibSVMBinaryReader<ElemType>::ReleaseMemory()
            {
            }

            template<class ElemType>
            bool LibSVMBinaryReader<ElemType>::Randomize()
            {
                if (m_randomize > 0)
                {
                    return true;
                }
                return false;
            }


            template<class ElemType>
            void LibSVMBinaryReader<ElemType>::Shuffle(size_t lowerBound, size_t windowSize)
            {
				FillReadOrder(lowerBound, windowSize);
                if (Randomize())
                {
                    int useLast = (m_partialMinibatch) ? 1 : 0;
                    shuffle(&read_order[0], &read_order[m_windowSize - useLast], random_engine);
                }
            }

            template<class ElemType>
            bool LibSVMBinaryReader<ElemType>::GetData(const std::wstring&, size_t, void*, size_t&, size_t)
            {
                throw runtime_error("GetData not supported in LibSVMBinaryReader");
            }

            // instantiate all the combinations we expect to be used
            template class LibSVMBinaryReader<double>;
            template class LibSVMBinaryReader<float>;
        }
    }
}
