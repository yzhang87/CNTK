// DataReaderHelper.h -- helper functions that understand both DataReader and ComputationNetwork

#pragma once

#include "Basics.h"
#include "DataReader.h"
#include "ComputationNetwork.h"
#include "MPIWrapper.h"
#include <string>
#include <map>
#include "TrainingCriterionNodes.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    /*static*/ struct DataReaderHelpers
    {
        // -------------------------------------------------------------------
        // DecimateMinibatch - decimate minibatch for parallelization
        // -------------------------------------------------------------------

        // We sub-sample the parallel utterances.
        template<class ElemType>
        static void DecimateMinibatch(std::map<std::wstring, Matrix<ElemType>*> &mb,    // matrix to be decimated
                                      int numprocs, int rank,                           // rank info
                                      MBLayoutPtr pMBLayout)                            // gets decimated as well
        {
            if (numprocs == 1)
                return;

            // For RNN, a input Matrix is organized in the following way:
            //   | x_t^1  x_t^2 ... x_t^N |  .... | x_{t+T-1}^1 ... x_{t+T-1}^N |
            //   |<----   block 1    ---->|  .... |<------  block T       ----->|
            // N is the nSlice (input)
            // The decimation here is to split each block to individual GPUs
            // So After decimation
            //   | x_t^{st} ... x_t^{en-1}|  .... | x_{t+T-1}^{st} ... x_{t+T-1}^{en-1} |
            // Each block now has nSlice/nProcs
            //
            // Correspondingly, the MBLayout will be revised

            size_t nOrigParallelUtts = pMBLayout->GetNumParallelSequences();
            size_t T = pMBLayout->GetNumTimeSteps();

            // decide new parallel utterances
            size_t sent_start = nOrigParallelUtts * (size_t)rank / numprocs;
            size_t sent_end = nOrigParallelUtts * (size_t)(rank + 1) / numprocs;
            static bool warned = false;
            if (nOrigParallelUtts % numprocs != 0 && !warned)
            {
                /* give a warning of potential bandwidth wasting */
                fprintf(stderr, "DecimateMinibatch: WARNING: Number of parallel utterances %d not a multiple of number of GPUs %d, GPU usage will be suboptimal.\n",
                        (int)nOrigParallelUtts, (int)numprocs);
                warned = true;
            }
            size_t newNumParallelSequences = sent_end - sent_start;

            // decimate data
            size_t rv = 0;
            for (auto & it : mb)
            {
                MSR::CNTK::Matrix<ElemType> &mat = *it.second;
                size_t nCols = mat.GetNumCols();

                // assert the cols are even among nodes
                if (rv == 0)
                    rv = nCols;
                else if (rv != nCols)
                    LogicError("DecimateMinibatch: Inconsistent number of columns among inputs (found %d and %d).", (int)rv, (int)nCols);

                if (T != nCols / nOrigParallelUtts)
                    LogicError("ERROR: MBLayout borked, GetNumTimeSteps() mismatches minibatch number of columns\n");
                if (T * nOrigParallelUtts != nCols) // (should really not happen)
                    LogicError("ERROR: minibatch size %d, but with %d parallel utterances --layout information borked\n", nCols, nOrigParallelUtts);

                if (sent_end == sent_start)
                {
                    // should never happen, print debug info
                    // BUGBUG: Yes, this can happen if we got less parallel sequences than GPUs. But users wouldn't want that, so we should fail.
                    // BUGBUG: This can also happen for a very small minibatch at the end of the epoch.
                    fprintf(stderr, "DecimateMinibatch: WARNING: col_st=col_en=%d, nCol=%d, nBlock=%d, nParaUtts=%d, nGPU=%d--This can happen if #parallel sequences < #GPUs (you'd be leaving a GPU unused)\n",
                        (int)sent_start, (int)nCols, (int)T, (int)nOrigParallelUtts, (int)numprocs);
                }

                // copy the respective columns
                // TODO: not efficient. Instead, use Reshape() and AssignRowSlice...()
                MSR::CNTK::Matrix<ElemType> tmp(mat.GetNumRows(), newNumParallelSequences*T, mat.GetPreferredDeviceId(), mat.GetMatrixType());
                for (size_t t = 0; t < T; t++)
                    tmp.SetColumnSlice(mat.ColumnSlice(t*nOrigParallelUtts + sent_start, newNumParallelSequences), t*newNumParallelSequences, newNumParallelSequences);
                mat.SetValue(tmp);      // update matrix in-place (new matrix has less parallel streams)
                // TODO: ^^ If had Matrix::RowSlice(), this would be simpler.
                //       TODO: But we do have a row-slice assignment function. This could be used.
            }
            // decimate layout
            auto pNewMBLayout = make_shared<MBLayout>(newNumParallelSequences, T, true);
            for (size_t t = 0; t < T; t++) for (size_t id = 0; id < newNumParallelSequences; id++)
                pNewMBLayout->Set(id, t, pMBLayout->Get(id + sent_start, t));
            pMBLayout->MoveFrom(pNewMBLayout);  // update layout in-place
        }

        static bool AtEndOfEpoch(bool useDistributedMBReading, bool wasDataRead)
        {
            // TODO mahilleb: reader should be able to return this information without an MPI AllReduce?
            if (useDistributedMBReading)
            {
                // In case of distributed reading, the current node needs to continue even with a minibatch size of 0 if any
                // other node in the group has a non-zero size minibatch to process. This is needed to ensure that
                // the gradient aggregation barriers do not get stuck and also to ensure that all nodes update their weights
                // properly using the aggregate gradients from other nodes before moving on to the next epoch even though the current
                // node itself may not have any gradient contribution.
                // TODO: wasDataRead == false means end of epoch, right? Is this state idempotent?
                std::array<int, 1> numNodesWithDataToProcess;
                numNodesWithDataToProcess[0] = wasDataRead ? 1 : 0;
                g_mpi->AllReduce(numNodesWithDataToProcess);

                return numNodesWithDataToProcess[0] == 0;
            }
            return !wasDataRead;
        }

        // -------------------------------------------------------------------
        // GetMinibatchIntoNetwork() -- get one minibatch from Reader (this->trainSetDataReader) into Network (this->net)
        // Returns false if end of epoch has been reached.
        // If not, then actualMBSize is set. Note that 0 is a valid value to be returned for actualMBSize, caller must handle that correctly.
        // -------------------------------------------------------------------

        // Note: Later, a function like this will become part of the reader interface.
        // TODO: callers of this often do ComputationNetwork::UpdateEvalTimeStamps(featureNodes) and also for labels; we should eliminate the need for this.
        template<class ElemType>
        static bool GetMinibatchIntoNetwork(IDataReader<ElemType>& trainSetDataReader,
                                            ComputationNetwork& net,
                                            ComputationNodeBasePtr criterionNode,
                                            bool useDistributedMBReading,
                                            bool useParallelTrain,
                                            std::map<std::wstring, Matrix<ElemType>*> & inputMatrices,
                                            size_t & actualMBSize)
        {
            // Get a new minibatch, filling in the matrices of the input nodes and the minibatch layout directly.
            bool wasDataRead = trainSetDataReader.GetMinibatch(inputMatrices, net.GetMBLayoutPtr());

            if (criterionNode != nullptr)
            {
                // eldak: processing unit should be taken from getminibatch.
                // mahilleb: need to check if this already requires NotifyInputNodesFunctionValuesMBSizeModified();
                criterionNode->UpdateWithMinibatch(ProcessingUnit());
            }

            // Update the network to reflect potentially updated number of columns of the matrices of the input nodes.
            net.NotifyInputNodesFunctionValuesMBSizeModified();

            // Determine if all feature nodes' input matrices have zero columns now;
            // this is (currently) a special case of the reader returning no data to process.
            // TODO should get rid of this in new reader interface? (Instead reader might return end-of-epoch information)
            actualMBSize = wasDataRead ? net.DetermineActualMBSizeFromFeatures() : 0;
            if (actualMBSize == 0)
            {
                wasDataRead = false;
            }

            // Check (and return) if at the end of an epoch.
            if (AtEndOfEpoch(useDistributedMBReading, wasDataRead))
            {
                return false;
            }

            // We are not at the end of epoch.
            // Note, however, that in case of parallelization, this worker may have received a share of 0 samples. Calling code, beware.

            if (wasDataRead && !useDistributedMBReading && useParallelTrain)
            {
                assert(0); // we don't do this yet .. do we need it or should be always done by reader?

                // Decimate (in-place).
                DecimateMinibatch(inputMatrices, g_mpi->NumNodesInUse(), g_mpi->CurrentNodeRank(), net.GetMBLayoutPtr());

                // Reflect reduced column sizes.
                net.NotifyInputNodesFunctionValuesMBSizeModified();

                // Recompute minibatch size (note: might be 0 now)
                actualMBSize = net.DetermineActualMBSizeFromFeatures();
            }

            // left-over TODOs:
            // TODO: This should be called/callable outside if 'wasDataRead' (GetMinibatch() should fill matrices to empty)
            // TODO: This will go away, as we will do resizing inside EvaluateThisNode(FrameRange()).

            // Prepare the network for processing the next minibatch
            // (TODO some of the above functionality should be moved in there)
            net.PrepareNewMinibatch();

            return true;
        }
    };

}}}
