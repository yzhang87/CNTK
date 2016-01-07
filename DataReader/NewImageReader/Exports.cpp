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

namespace Microsoft { namespace MSR { namespace CNTK {

extern "C" DATAREADER_API void GetReaderF(IDataReader<float>** preader)
{
    GetReader(preader);
}

extern "C" DATAREADER_API void GetReaderD(IDataReader<double>** preader)
{
    GetReader(preader);
}

template<class ElemType>
void DATAREADER_API GetReader(IDataReader<ElemType>** preader)
{
    auto factory = [](const ConfigParameters& parameters) -> ReaderPtr { return std::make_shared<ImageReader>(parameters, sizeof(ElemType)); };
    *preader = new ReaderShim<ElemType>(factory);
}

}}}
