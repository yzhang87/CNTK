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
#include "ImageReader.h"
#include "ReaderShim.h"
#include "ImageReaderNew.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
void DATAREADER_API GetReader(IDataReader<ElemType>** preader)
{
    *preader = new ImageReader<ElemType>();
}

extern "C" DATAREADER_API void GetReaderF(IDataReader<float>** preader)
{
    GetReader(preader);
}
extern "C" DATAREADER_API void GetReaderD(IDataReader<double>** preader)
{
    GetReader(preader);
}

ReaderPtr CreateReader(const ConfigParameters& parameters)
{
    return std::make_shared<ImageReaderNew>(parameters, nullptr);
}

template<class ElemType>
void DATAREADER_API GetReaderNew(IDataReader<ElemType>** preader)
{
    *preader = new ReaderShim<ElemType>(CreateReader);
}

extern "C" DATAREADER_API void GetReaderFNew(IDataReader<float>** preader)
{
    GetReaderNew(preader);
}
extern "C" DATAREADER_API void GetReaderDNew(IDataReader<double>** preader)
{
    GetReaderNew(preader);
}

}}}