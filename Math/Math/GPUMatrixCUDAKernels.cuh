//
// <copyright file="GPUMatrixCUDAKernels.cuh" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "BestGpu.h"

#ifndef CPUONLY

#include <float.h>
#include <cuda_runtime.h>
#include "CommonMatrix.h"
#include "device_functions.h"
#include <assert.h>

// REVIEW alexeyk: disable warnings properly for GCC/clang
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
#pragma warning (disable: 4201)
#pragma warning (disable: 4515)
#endif
#include <cub/cub.cuh>
#ifdef _MSC_VER
#pragma warning (pop)
#endif

// We would like to use 64-bit integer to support large matrices. However, CUDA seems to support only 32-bit integer
// For now, use int32_t to ensure that both Linux and Windows see this as 32 bit integer type.

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

#define threadsPerBlock 512

#ifdef __GNUC__
#define UNUSED_FUNCTION_ATTRIBUTE __attribute__ ((unused))
#else
#define UNUSED_FUNCTION_ATTRIBUTE
#endif

////CUDA Kernels code
template<class ElemType>
__global__ void _elementWisePowerOnCuda(
    ElemType alpha,
    const ElemType *a,
    ElemType* c,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _inplaceSigmoidOnCuda(
    ElemType* c,
    const CUDA_LONG N);

//__device__ __forceinline__ float _exp(float f)
//{
//    return expf(f);
//}
//
//__device__ __forceinline__ double _exp(double f)
//{
//    return exp(f);
//}
//

template<class ElemType>
__global__ void _assignSigmoidOf(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _inplaceLinRectDerivative(
    ElemType* c,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _assignSigmoidDerivative(
    ElemType *a,
    ElemType *c,
    CUDA_LONG N);

template<class ElemType>
__global__ void _inplaceTanhOnCuda(
    ElemType* c,
    const CUDA_LONG N);

////to prevent negative values caused by floating operations, we force inputs to be >=0
////this may, however, hide problems in the caller.
template<class ElemType>
__global__ void _inplaceSqrtOnCuda(
    ElemType* c,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _inplaceExpOnCuda(
    ElemType* c,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _inplaceLogOnCuda(
    ElemType* c,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _inplaceAbsOnCuda(
    ElemType* c,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _inplaceCosineOnCuda(
    ElemType* c,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _inplaceNegativeSineOnCuda(
    ElemType* c,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _setValue(
    ElemType* a,
    const ElemType v,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _setValue(
    ElemType* a,
    const ElemType* d_v,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _copyColumnsStrided(ElemType * dest, ElemType * src, CUDA_LONG N, CUDA_LONG numRows, CUDA_LONG destNumColsStride, CUDA_LONG srcNumColsStride);

template<class ElemType>
__global__ void _assignToRowSliceValuesOf(ElemType * dest, ElemType * src, const CUDA_LONG N, const CUDA_LONG startIndex, const CUDA_LONG destRows, const CUDA_LONG srcRows);

template<class ElemType>
__global__ void _assignRowSliceValuesOf(ElemType * dest, ElemType * src, const CUDA_LONG N, const CUDA_LONG startIndex, const CUDA_LONG destRows, const CUDA_LONG srcRows);

template<class ElemType>
__global__ void _addToRowSliceValuesOf(ElemType * dest, ElemType * src, const CUDA_LONG N, const CUDA_LONG startIndex, const CUDA_LONG destRows, const CUDA_LONG srcRows);

template<class ElemType>
__global__ void _addWithRowSliceValuesOf(ElemType * dest, ElemType * src, const CUDA_LONG N, const CUDA_LONG startIndex, const CUDA_LONG destRows, const CUDA_LONG srcRows);

template<class ElemType>
__global__ void _assignToDiagonalValuesOf(ElemType * dest, ElemType * src, const CUDA_LONG N, const CUDA_LONG srcCols);

template<class ElemType>
__global__ void _assignRowStackValuesOf(ElemType * dest, ElemType ** srces, size_t* startRowIndeces, const CUDA_LONG numSrces, const CUDA_LONG N, const CUDA_LONG destRows, const CUDA_LONG destCols);

template<class ElemType>
__global__ void _assignRepeatOf(ElemType * dest, ElemType * src, const CUDA_LONG N, const CUDA_LONG srcRows, const CUDA_LONG srcCols, const CUDA_LONG destRows);

template<class ElemType>
__global__ void _addToRowRepeatValuesOf(ElemType * dest, ElemType * src, const CUDA_LONG N, const CUDA_LONG srcRows, const CUDA_LONG srcCols, const CUDA_LONG destRows);

template<class ElemType>
__global__ void _assignPositiveAndShiftedNegSample(ElemType * dest, const ElemType * src, const CUDA_LONG N, const CUDA_LONG srcRows, const CUDA_LONG srcCols, const CUDA_LONG destRows, const CUDA_LONG posNumber, const CUDA_LONG shiftNumber);

template<class ElemType>
__global__ void _addFoldedPositiveAndShiftedNegSample(ElemType * folded, const ElemType * unfolded, const CUDA_LONG unfoldedN, const CUDA_LONG unfoldedRows, const CUDA_LONG unfoldedCols, const CUDA_LONG foldedRows, const CUDA_LONG posNumber, const CUDA_LONG shiftNumber);

template<class ElemType>
__global__ void _assignDifferenceOf1(
    ElemType* us,
    const ElemType alpha,
    const ElemType* a,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _assignDifferenceOf2(
    ElemType* us,
    const ElemType alpha,
    const ElemType* a,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _scaleAndAddScalar(
    ElemType* c,
    const CUDA_LONG N,
    const ElemType alpha,
    const ElemType* a
    );

template<class ElemType>
__global__ void _multiply1x1AndWeightedAdd(
    ElemType alpha, const ElemType* a, const ElemType* b, ElemType beta, ElemType* c, CUDA_LONG N);

template<class ElemType>
__global__ void _addValue(
    ElemType* a,
    const ElemType v,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _addValue(
    ElemType* a,
    const ElemType* d_v,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _elemMul(
    ElemType* a,
    const ElemType* b,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _assignElementProductOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _assignKhatriRaoProductOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG rowsA,
    const CUDA_LONG rowsB,
    const CUDA_LONG cols);

template<class ElemType>
__global__ void _addColumnReshapeProductOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG rowsB,
    const CUDA_LONG rowsC,
    const CUDA_LONG cols,
    const bool transposeAColumn);

template<class ElemType>
__global__ void _columnElementMultiplyWith(
    ElemType* us,
    const ElemType* a,
    const CUDA_LONG N, //a.GetNumRows();
    const CUDA_LONG M); //us.GetNumCols();

template<class ElemType>
__global__ void _rowElementMultiplyWith(
    ElemType* us,
    const ElemType* a,
    const CUDA_LONG N, //us.GetNumRows();
    const CUDA_LONG M); //a.GetNumCols();

template<class ElemType>
__global__ void _assignElementDivisionOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _elemInverse(
    ElemType* us,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _logSoftMaxColWise(
    ElemType *a,
    const CUDA_LONG m_numCols,
    const CUDA_LONG m_numRows);

// each block processes one column. There must be 512 threads in a block
template<class ElemType>
__global__ void _assignColumnwiseLogSoftmaxOf(
    const ElemType *a,
    ElemType* us,
    const CUDA_LONG m_numCols,
    const CUDA_LONG m_numRows);

template<class ElemType>
__global__ void _logSoftMaxRowWise(
    ElemType *a,
    const CUDA_LONG m_numCols,
    const CUDA_LONG m_numRows); //ld

// each block processes one column. There must be 512 threads in a block
template<class ElemType>
__global__ void _assignColumnwiseHardmaxOf(
    const ElemType *a,
    ElemType* us,
    const CUDA_LONG m_numCols,
    const CUDA_LONG m_numRows);

template<class ElemType>
__global__ void _inplaceTruncateBottom(
    ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _assignTruncateBottom(
    ElemType* us,
    const ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _inplaceTruncateTop(
    ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _assignTruncateTop(
    ElemType* us,
    const ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _setToZeroIfAbsLessThan(
    ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _areEqual(
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N,
    const ElemType threshold,
    long *d_res);

// see Matrix<ElemType>::TensorShuffleScaleAndAdd() for comments
template<class ElemType>
__global__ void _tensorShuffleScaleAndAdd(
    ElemType keepWeight, const ElemType* pa, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const ElemType* pb, ElemType* pc);

template<class ElemType>
__global__ void _hasElement(
    const ElemType* a,
    const CUDA_LONG N,
    ElemType *d_res  /// [2x1] vector. The first is the value to be compared and the second is the 0/1 to return
    );

template<class ElemType>
__global__ void _setDiagonalValue(
    ElemType* a,
    const ElemType v,
    const CUDA_LONG N,
    const CUDA_LONG ld);

template<class ElemType>
__global__ void _setDiagonalValueFromVector(
    ElemType* a,
    const ElemType* b,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _adagrad(
    ElemType* a,
    ElemType* d_v,
    const CUDA_LONG N,
    ElemType* multipliers);

template<class ElemType>
__global__ void _adagrad4BlockSparse(
    ElemType* a,  //dense
    const size_t numRows, //number of rows in a and in d_v
    ElemType* d_v, //block sparse
    const GPUSPARSE_INDEX_TYPE* blockId2ColOrRow,
    ElemType* multipliers,
    const bool colMajor,
    const size_t len, //major dim, numRows in colMajor and numcols in rowMajor
    const CUDA_LONG N); //total number of non-zero values

template<class ElemType>
__global__ void _fsadagrad(CUDA_LONG size, ElemType* grad, ElemType* smoothAda, ElemType* smoothMom, ElemType* val,
    ElemType lr, ElemType mom, ElemType adaWeight, ElemType adaMul);

template<class ElemType>
__global__ void _rmsprop_init(
    ElemType* avars, ElemType* signs, ElemType* steps,
    ElemType* curr_grad,
    const CUDA_LONG N
    );

template<class ElemType>
__global__ void _rmsprop(
    ElemType* avars, ElemType* signs, ElemType* steps,
    ElemType* curr_grad,
    const CUDA_LONG N,
    ElemType RMS_GAMMA, ElemType RMS_WGT_INC, ElemType RMS_WGT_MAX, ElemType RMS_WGT_DEC, ElemType RMS_WGT_MIN,
    ElemType floor,
    ElemType *upd_gpu,
    ElemType* multipliers
    );
template<class ElemType>
__global__ void _rescaleToRange(
    ElemType* a,
    const CUDA_LONG N,
    const ElemType low,
    const ElemType high);

template<class ElemType>
__global__ void _setMaskAndScale(
    ElemType* a,
    const CUDA_LONG N,
    const ElemType maskRate,
    const ElemType scaleValue);

template<class ElemType>
__global__ void _vectorSum(
    ElemType* c, //output
    const ElemType* a, //input
    const CUDA_LONG n, //a.numRows
    const CUDA_LONG m, //a.numCols
    const bool isColWise);

template<class ElemType>
__global__ void _vectorNorm1(
    ElemType* c, //output
    const ElemType* a, //input
    const CUDA_LONG n, //a.numRows
    const CUDA_LONG m, //a.numCols
    const bool isColWise);

//one column per thread
template<class ElemType>
__global__ void _vectorNorm2(
    ElemType* c,  //output
    const ElemType* a, //input
    const CUDA_LONG N, //a.GetNumRows();
    const CUDA_LONG M, //a.GetNumCols();
    const bool isColWise);

template<class ElemType>
__global__ void _convertInd2ValsAdjustInd(
    ElemType* inds,
    const ElemType* M,
    ElemType* vals,
    const CUDA_LONG n, //number of cols
    const CUDA_LONG m, //number of rows
    const bool isColWise);

//assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
template<class ElemType>
__global__ void _assignPackedConvolutionInput(ElemType * packedMatrix, const ElemType * inputSubBatch, const CUDA_LONG batchSize,
    const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputChannels,
    const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputChannels,
    const CUDA_LONG kernelWidth, const CUDA_LONG kernelHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample, const bool zeroPadding);

//assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
template<class ElemType>
__global__ void _unpackConvolutionInput(const ElemType * packedMatrix, ElemType * inputSubBatch, const CUDA_LONG batchSize,
    const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputChannels,
    const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputChannels,
    const CUDA_LONG kernelWidth, const CUDA_LONG kernelHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample, const bool zeroPadding);

template<class ElemType>
__global__ void _assignMaxPoolingResult(ElemType * outputBatch, const ElemType * inputBatch, const CUDA_LONG batchSize, const CUDA_LONG channels,
    const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputSizePerSample,
    const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputSizePerSample,
    const CUDA_LONG windowWidth, const CUDA_LONG windowHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample);

template<class ElemType>
__global__ void _addMaxPoolingGradient(ElemType * inputGradientBatch, const ElemType * outputGradientBatch, const ElemType * inputBatch, const ElemType * outputBatch,
    const CUDA_LONG batchSize, const CUDA_LONG channels,
    const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputSizePerSample,
    const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputSizePerSample,
    const CUDA_LONG windowWidth, const CUDA_LONG windowHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample);

template<class ElemType>
__global__ void _assignAveragePoolingResult(ElemType * outputBatch, const ElemType * inputBatch, const CUDA_LONG batchSize, const CUDA_LONG channels,
    const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputSizePerSample,
    const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputSizePerSample,
    const CUDA_LONG windowWidth, const CUDA_LONG windowHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample);

template<class ElemType>
__global__ void _addAveragePoolingGradient(ElemType * inputGradientBatch, const ElemType * outputGradientBatch,
    const CUDA_LONG batchSize, const CUDA_LONG channels,
    const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputSizePerSample,
    const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputSizePerSample,
    const CUDA_LONG windowWidth, const CUDA_LONG windowHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample);

template<class ElemType>
__global__ void _addMaxPoolingGradientLoopOut(ElemType * inputGradientBatch, const ElemType * outputGradientBatch, const ElemType * inputBatch, const ElemType * outputBatch,
    const CUDA_LONG batchSize, const CUDA_LONG channels,
    const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputSizePerSample,
    const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputSizePerSample,
    const CUDA_LONG windowWidth, const CUDA_LONG windowHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample);

template<class ElemType>
__global__ void _addElementProductOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N);

template<class elemtype>
__global__ void _rowelementmultiplywith(
    elemtype* us,
    const elemtype* a,
    const CUDA_LONG n, //us.getnumrows();
    const CUDA_LONG m); //a.getnumcols();

template<class ElemType>
__global__ void _rowElementDivideBy(
    ElemType* us,
    const ElemType* a,
    const CUDA_LONG N, //us.GetNumRows();
    const CUDA_LONG M); //a.GetNumCols();

template<class ElemType>
__global__ void _ColumnElementDivideBy(
    ElemType* us,
    const ElemType* a,
    const CUDA_LONG N, //a.GetNumRows();
    const CUDA_LONG M); //us.GetNumCols();

template<class ElemType>
__global__ void _innerProduct(
    ElemType* c,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N, //a.GetNumRows();
    const CUDA_LONG M, //a.GetNumCols();
    const bool isColWise);

template<class ElemType>
__global__ void _assignSignOf(
    ElemType* a,
    const ElemType* b,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _addSignOf(
    ElemType* a,
    const ElemType* b,
    const CUDA_LONG N);

// This function processes 1 column per block. this function needs 512 threads
template<class ElemType, bool IsMax>
__global__ void _vectorMaxMinReduce(
    const ElemType* us,
    ElemType* Indexes,
    ElemType* Values,
    const CUDA_LONG numRows,
    const CUDA_LONG numCols);

template<class ElemType>
__global__ void _vectorMax(
    const ElemType* us,
    ElemType* maxIndexes,
    ElemType* maxValues,
    const CUDA_LONG m,  //number of rows
    const CUDA_LONG n,  //number of cols
    const bool isColWise);

template<class ElemType>
__global__ void _vectorMin(
    const ElemType* us,
    ElemType* minIndexes,
    ElemType* minValues,
    const CUDA_LONG m,  //number of rows
    const CUDA_LONG n,  //number of cols
    const bool isColWise);

//this implementation uses more threads but also more memory access
template<class ElemType>
__global__ void _matrixVectorColumnWiseAddWithThreadPerElem(
    const ElemType* a,
    ElemType* us,
    ElemType alpha,
    const CUDA_LONG m,  //number of rows
    const CUDA_LONG n);  //number of cols     

template<class ElemType>
__global__ void _matrixVectorColumnWiseAddWithThreadPerRow(
    const ElemType* a,
    ElemType* us,
    ElemType alpha,
    const CUDA_LONG m,  //number of rows
    const CUDA_LONG n);  //number of cols     

template<class ElemType>
__global__ void _matrixVectorColumnWiseAddBlockPerRow(
    const ElemType* a,
    ElemType* us,
    ElemType alpha,
    const CUDA_LONG m,  //number of rows
    const CUDA_LONG n);  //number of cols     

template<class ElemType>
__global__ void _addScaledDifference(
    ElemType alpha,
    ElemType *a,
    ElemType *b,
    ElemType *c,
    CUDA_LONG N);

template<class ElemType>
__global__ void _assignScaledDifference(
    ElemType alpha,
    ElemType *a,
    ElemType *b,
    ElemType *c,
    CUDA_LONG N);

template<class ElemType>
__global__ void _addScaledDifference(
    ElemType *alpha,
    ElemType *a,
    ElemType *b,
    ElemType *c,
    CUDA_LONG N);

template<class ElemType>
__global__ void _assignScaledDifference(
    ElemType *alpha,
    ElemType *a,
    ElemType *b,
    ElemType *c,
    CUDA_LONG N);

template<class ElemType>
__global__ void _addElementToElement(
    const ElemType *a, CUDA_LONG indexA,
    ElemType *c, CUDA_LONG indexC);

template<class ElemType>
__global__ void _assignNumOfDiff(
    const ElemType *a,
    const ElemType *b,
    ElemType *c,
    CUDA_LONG N);

template<class ElemType>
__global__ void _scaleArray(
    ElemType alpha,
    ElemType *us,
    CUDA_LONG N);

template<class ElemType>
__global__ void _sparseCSRPlusDense(
    ElemType alpha,
    const ElemType* m_dVal,
    const int* m_dRow,
    const int* m_dCol,
    ElemType* pArrayDev,
    CUDA_LONG M);

template<class ElemType>
__global__ void _sparseCSRElemMulDense(
    const ElemType* m_dVal,
    const int* m_dRow,
    const int* m_dCol,
    const ElemType* b,
    ElemType* c,
    CUDA_LONG M);

//c = alpha * op(a) * op(b) + beta*c
//this function can be further improved by using shared memory
template<class ElemType>
__global__ void _denseMultSparseCSCAndWeightedAddToDense(
    int m, //rowDense
    int n,   //colSparse
    ElemType alpha,
    const ElemType* a,  //dense
    const ElemType* bnzValues,  //sparse nz values
    const GPUSPARSE_INDEX_TYPE* rowIndex,
    const GPUSPARSE_INDEX_TYPE* colCSCIndex,
    ElemType beta,
    ElemType* c  //dense target
    );

/// c += alpha * a * b^T
template<class ElemType>
__global__ void _denseMultSparseCSCTransposeAndAddToDense(
    int m, //rowDense
    int n,   //number of columns in sparse matrix
    int colInC, /// column index of the sparse matrix
    ElemType alpha,
    const ElemType* a,  //dense
    const ElemType* bnzValues,  //sparse nz values
    const GPUSPARSE_INDEX_TYPE* rowIndex,
    const GPUSPARSE_INDEX_TYPE* colCSCIndex,
    ElemType* c  //dense target
    );

//called before _determineBlockIds and _denseMulSparseCSCTransposeToSparseBlockCol to determine which columns have values and
//what's the mapping from the column id in the resulted SparseBlockCol format to the column id in the dense format
//input: rowIndexes: the row indexes of the CSC sparse matrix to be multiplied with
//blockIDs: the blockID mapping in the resulting matrix; 
//nnz: number of nonzero value or the size of rowIndexes;
template<class ElemType>
__global__ void _findColsWithValues(
    const GPUSPARSE_INDEX_TYPE* rowIndexes, GPUSPARSE_INDEX_TYPE* blockIds, const size_t nnz);

//called before _denseMulSparseCSCTransposeToSparseBlockCol and after _findColsWithValuesto determine which columns have values and
//what's the mapping from the column id in the resulted SparseBlockCol format to the column id in the dense format
//input: rowIndexes: the row indexes of the CSC sparse matrix to be multiplied with
//blockId2Col: the blockID to colum id mapping in the resulting matrix; 
//col2BlockId: the col2BlockId to blockID mapping in the resulting matrix; 
//numCols: number of columns in the resulting matrix or the size of blockIDs
//blockSize: return the blockSize with values, *blockSize must be zero before passed in.
template<class ElemType>
__global__ void _determineBlockIds(
    GPUSPARSE_INDEX_TYPE* blockId2Col, GPUSPARSE_INDEX_TYPE*col2BlockId, const size_t numCols, size_t* blockSize);

// backward pass from hidden layer to feature weight
//result (sparse BlockCol)= alpha * (lhs (dense) X rhs^T (sparse CSC)
//assume resultValues are 0-initialized
template<class ElemType>
__global__ void _denseMulSparseCSCTransposeToSparseBlockCol2(
    const ElemType alpha,
    const ElemType* lhsValues,
    const size_t numRowsLhs,
    const size_t numColsRhs,
    const ElemType* rhsNZValues,
    const GPUSPARSE_INDEX_TYPE* rhsRows,
    const GPUSPARSE_INDEX_TYPE* rhsCols,
    const GPUSPARSE_INDEX_TYPE* col2blockIds,
    ElemType* resultValues);

// backward pass from hidden layer to feature weight
//result (sparse BlockCol)= alpha * (lhs (dense) X rhs^T (sparse CSC)
//assume resultValues are 0-initialized
template<class ElemType>
__global__ void _denseMulSparseCSCTransposeToSparseBlockCol(
    const ElemType alpha,
    const ElemType* lhsValues,
    const size_t numRowsLhs,
    const size_t numColsRhs,
    const ElemType* rhsNZValues,
    const GPUSPARSE_INDEX_TYPE* rhsRows,
    const GPUSPARSE_INDEX_TYPE* rhsCols,
    const GPUSPARSE_INDEX_TYPE* rhsRowIdx,
    ElemType* resultValues,
    GPUSPARSE_INDEX_TYPE* resultBlockIds);

// gradients update
template<class ElemType>
__global__ void _scaleSparseBlockAndAddToDense(
    const ElemType alpha,
    const bool blockCol, //true if blockRow
    const size_t numRows,
    const size_t numCols,
    const size_t numBlocks,
    const ElemType* lhsValues,  //lhs is blockCol or blockRow
    const GPUSPARSE_INDEX_TYPE* blockIds,
    ElemType* rhs);

// compute predictions in cross entory node
template<class ElemType>
__global__ void _computePrediction(
    int nv,
    const ElemType* a,
    int numrows,
    const ElemType* weight,
    int nrs,
    int labelSize,
    const GPUSPARSE_INDEX_TYPE* labelRow,
    const size_t* block2Id,
    const ElemType* cls,
    const ElemType* idx2cls,
    ElemType* val,
    GPUSPARSE_INDEX_TYPE* row,
    GPUSPARSE_INDEX_TYPE* pb);

// normalize predictions in cross entropy node
template<class ElemType>
__global__ void _normalizePrediction(
    const size_t labelSize,
    const size_t expandedLabelSize,
    const GPUSPARSE_INDEX_TYPE* labelRow,
    const size_t* block2Id,
    const GPUSPARSE_INDEX_TYPE* row,
    ElemType* val,
    ElemType* entropyScore);

// compute prediction error in cross entropy node
template<class ElemType>
__global__ void _computePredictionError(
    ElemType* val,
    int N);

// compute gradients of input in cross entropy node
template<class ElemType>
__global__ void _computeGradientOfInput(
    const ElemType* val,
    const GPUSPARSE_INDEX_TYPE* row,
    const GPUSPARSE_INDEX_TYPE* pb,
    ElemType* weight,
    size_t nrs,
    ElemType* grd,
    size_t numrows);

template<class ElemType>
__global__ void computeNCEForwardProp(
    const ElemType* val,
    const int* col,
    int numRows,
    int sampleCount,
    const ElemType* a,
    int numCols_a,
    const ElemType* b,
    ElemType* res);

template<class ElemType>
__global__ void _computeNceOutput(
    const ElemType* col,
    int numRows,
    int sampleCount,
    const ElemType* a,
    int numCols_a,
    const ElemType* b,
    const ElemType* bias,
    ElemType* res);

template<class ElemType>
__global__ void _assignSoftmaxSum(
    const ElemType* softmax,
    int sampleCount,
    const ElemType* a,
    ElemType* c); // run on 512 threads per block

template<class ElemType>
__global__ void _assignNoiseContrastiveEstimation(
    const ElemType* val,
    int numRows,
    int sampleCount,
    const ElemType* a,
    int width, // number of columns in a
    const ElemType* b,
    ElemType* tmp,
    ElemType* c); // run on 512 threads per block

template<class ElemType>
__global__ void _assignNceDerivative(
    const ElemType* val,
    int numRows,
    int sampleCount,
    const ElemType* a,
    int width, // number of columns in a
    const ElemType* b,
    const ElemType* tmp,
    ElemType* c,
    size_t inputIndex);

template<class ElemType>
__global__ void _assignNceDerivativeNew(
    const ElemType* val,
    int numRows,
    int sampleCount,
    const ElemType* a,
    int width, // number of columns in a
    const ElemType* b,
    const ElemType* tmp,
    ElemType* c,
    size_t inputIndex);

// compute gradients of weights in cross entropy node
template<class ElemType>
__global__ void _computeGradientOfWeight(
    const ElemType* val,
    const GPUSPARSE_INDEX_TYPE* row,
    const GPUSPARSE_INDEX_TYPE* pb,
    size_t mb,
    size_t nv,
    const GPUSPARSE_INDEX_TYPE* labelRow,
    const size_t* labelBlock2UniqId,
    const ElemType* cls,
    const ElemType* idx2cls,
    ElemType* input,
    size_t nrs,
    ElemType* blockVal,
    GPUSPARSE_INDEX_TYPE* blockIds);

// used in clipping gradients
template<class ElemType>
__global__ void _inplaceTruncate(
    ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _inplaceSoftThreshold(
    ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N);


template<class ElemType>
__global__ void _normalGradForSparseBlock(
    const ElemType momentum,
    const bool blockCol, //true if blockRow
    const size_t numRows,
    const size_t numCols,
    const size_t numBlocks,
    ElemType* lhsValues,  //lhs is blockCol or blockRow
    const GPUSPARSE_INDEX_TYPE* blockIds,
    ElemType* rhs);

//static __inline__ __device__ double atomicAdd(double* address, double val)
//{
//    unsigned long long int* address_as_ull = (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
//    } while (assumed != old);
//
//    return __longlong_as_double(old);
//}
//
//template<class ElemType>
//static __inline__ __device__ ElemType logadd(ElemType x, ElemType y)
//{
//    ElemType temp, diff, z;
//
//    if (x < y)
//    {
//        temp = x; x = y; y = temp;
//    }
//    diff = y - x;
//    if (diff < MINLOGEXP)
//    {
//        return (x < LSMALL) ? LZERO : x;
//    }
//    else
//    {
//        z = exp(diff);
//        return x + log(1.0 + z);
//    }
//}
//
//This function should be called with 1024 threads per block and 1 block
//THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION!!!
template<class ElemType>
__global__ void _reductionSum(
    const ElemType* data,
    ElemType *sum,
    CUDA_LONG N);

//This function should be called with 1024 threads per block and 1 block
//THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION!!!
template<class ElemType>
__global__ void _reductionSumAndAssign(
    ElemType* toAssign,
    const ElemType* data,
    CUDA_LONG N, //length of data
    CUDA_LONG M); //length of toAssign

//This function should be called with 1024 threads per block and 1 block
//THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION!!!
template<class ElemType>
__global__ void _reductionSum2(
    const ElemType* data,
    ElemType *sum,
    CUDA_LONG N,
    bool takeSqrt = false);

//This function should be called with 1024 threads per block and 1 block
//THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION!!!
template<class ElemType>
__global__ void _reductionMatrixNormInf(
    const ElemType* data,
    ElemType *maxAbs,
    CUDA_LONG N);

//This function should be called with 1024 threads per block and 1 block
//THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION!!!
template<class ElemType>
__global__ void _reductionMatrixNorm0(
    const ElemType* data,
    ElemType *nz,
    CUDA_LONG N);

template<class ElemType>
__global__ void _getSparseVectorRepresntationForCSCMatrix(
    const int* m_dRow,
    const int* m_dCol,
    int* vectArray,
    const CUDA_LONG M,
    const CUDA_LONG N);

template<class ElemType>
__global__ void _lrHelper(
    const ElemType* data1,
    const ElemType* data2,
    const CUDA_LONG N,
    ElemType* d_res);

template<class ElemType>
__global__ void _assignElementProductOfWithShiftNeg(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const int shift,
    const int NTPlusOne,
    const int BS);

template<class ElemType>
__global__ void _innerProductWithShiftNeg(
    ElemType* c,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N, //a.GetNumRows();
    const CUDA_LONG M, //a.GetNumCols();
    const CUDA_LONG shift,
    const CUDA_LONG NTPlusOne
    );

template<class ElemType>
__global__ void _getARowByIndex(
    ElemType* us,
    const ElemType* a,
    const int O, // a's rows
    const int P, // a's cols
    const int m // the m-th row of a
    );

template<class ElemType>
__global__ void _conductRowElementMultiplyWithShift(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const int O, // b's rows
    const int P, // b's cols
    const int shift,
    const bool isafixed);

template<class ElemType>
__global__ void _assignElementProductOfWithShift(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const int shift,
    const CUDA_LONG N);

///// minus 1 at a specific position
template<class ElemType>
__global__ void _minusOneAt(
    ElemType *c,
    CUDA_LONG position,
    CUDA_LONG N);

///// the kernel function for RCRF  backward computation
///// assume a column slice of input and output
template<class ElemType>
__global__ void _rcrfBackwardCompute(
    const size_t iNumPos,
    const ElemType* galpha,   /// column slice at current time t
    ElemType* gbeta,          /// column slices with [row, 2] at current time t for [
    const ElemType* gpair_scores,
    const size_t iNumLab, const int shift);

/// the kernel function for RCRF  backward computation
/// assume a column slice of input and output
template<class ElemType>
__global__ void _rcrfBackwardCompute(
    const size_t t, /// time position 
    const size_t iNumPos,
    const ElemType* galpha,   /// column slice at current time t
    ElemType* gbeta,          /// column slices with [row, 2] at current time t for [
    const ElemType* gzeta,          /// column slices with [row, 2] at current time t for [
    const ElemType* gpair_scores,   /// column slice at current time t
    const size_t iNumLab, const int shift);

/// $\zeta_t(j) = {\sum_k exp(\delta_{t-1}(k) + a_{kj}(t))}$.
template<class ElemType>
__global__ void _rcrfBackwardComputeZeta(
    const size_t t, /// time position 
    const size_t iNumPos,
    const ElemType* galpha,   /// column slice at current time t
    ElemType* gzeta,          /// column slices with [row, 2] at current time t for [
    const ElemType* gpair_scores,
    const size_t iNumLab, const int shift);

/// $\zeta_t(j) = {\sum_k exp(\delta_{t-1}(k) + a_{kj}(t))}$.
template<class ElemType>
__global__ void _rcrfTransGrdComputeZeta(
    const int t, /// time position 
    const size_t iNumPos,
    const ElemType* galpha,   /// column slice at current time t
    ElemType* gzeta,          /// column slices with [row, 2] at current time t for [
    const ElemType* gpair_scores,
    const size_t iNumLab,
    const size_t start_lbl,
    const int shift);

template<class ElemType>
__global__ void _rcrfTransGrdCompute(
    int t,
    const size_t start_lbl,
    const ElemType*   galpha,
    const ElemType* gbeta,
    const ElemType* gzeta,
    const ElemType* gpair_scores,
    const ElemType * lbls,
    ElemType* grd,
    const size_t iNumPos,
    const size_t iNumLab,
    const int shift);

template<class ElemType>
__global__ void _reductionLogAddSum(
    const ElemType* data,
    ElemType *sum,
    const size_t sum_size,
    CUDA_LONG N);

// set the value of certain columns to be zero
// the column is decided by threshhold value
// TODO: This kernel has very poor performace and needs to
// be optimized
template<class ElemType>
__global__ void _DropFrame(
    ElemType *a,
    const ElemType *label,
    const ElemType *gamma,
    const ElemType framedropthreshhold,
    const long m_numCols,
    const long m_numRows); //ld

template<class ElemType>
__global__ void _AssignSequenceError(const ElemType hsmoothingWeight, ElemType *error, const ElemType *label,
    const ElemType *dnnoutput, const ElemType *gamma, ElemType alpha, const long N);

template<class ElemType>
__global__ void _copyTopKResults(const uint64_t* indexes, const ElemType* values, ElemType* maxIndexes, ElemType* maxValues,
    CUDA_LONG crow, CUDA_LONG ccol, int topK);

template<int BlockSize, class ElemType>
__global__ void _assignNumOfDiffCol(const ElemType *a, const ElemType *b, ElemType *c, CUDA_LONG crowB, CUDA_LONG ccol);

template<class ElemType>
__global__ void _maskColumnsValue(ElemType *a, const char *columnsMask, CUDA_LONG numCols, CUDA_LONG numRows, ElemType val);

#endif // !CPUONLY
