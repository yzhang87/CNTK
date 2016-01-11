//
// <copyright file="Exports.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// Exports.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "ReaderShim.h"
#include "ImageReader.h"
#include "HeapMemoryProvider.h"
#include "CudaMemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: Memory provider should be injected by SGD.

extern "C" DATAREADER_API void GetReaderF(IDataReader<float>** preader)
{
    auto factory = [](const ConfigParameters& parameters) -> ReaderPtr 
    {
        return std::make_shared<ImageReader>(std::make_shared<HeapMemoryProvider>(), parameters, ElementType::et_float);
    };
    *preader = new ReaderShim<float>(factory);
}

extern "C" DATAREADER_API void GetReaderD(IDataReader<double>** preader)
{
    auto factory = [](const ConfigParameters& parameters) -> ReaderPtr
    {
        return std::make_shared<ImageReader>(std::make_shared<HeapMemoryProvider>(), parameters, ElementType::et_double);
    };
    *preader = new ReaderShim<double>(factory);
}

}}}
