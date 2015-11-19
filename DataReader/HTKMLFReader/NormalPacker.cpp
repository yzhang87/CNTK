#include "stdafx.h"
#include "NormalPacker.h"
#include "HtkMlfSource.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    template<class ElemType>
    NormalPacker<ElemType>::NormalPacker(/*std::shared_ptr<IMemoryProvider> provider, size_t minibatchSize, std::shared_ptr<ISequencer> source, */const ConfigParameters readerConfig)
        : Packer(std::shared_ptr<IMemoryProvider>(), 0, std::shared_ptr<ISequencer>())
    {
        ConfigArray numberOfuttsPerMinibatchForAllEpochs = readerConfig("nbruttsineachrecurrentiter", "1");
        m_numSeqsPerMBForAllEpochs = numberOfuttsPerMinibatchForAllEpochs;

        for (int i = 0; i < m_numSeqsPerMBForAllEpochs.size(); i++)
        {
            if (m_numSeqsPerMBForAllEpochs[i] < 1)
            {
                LogicError("nbrUttsInEachRecurrentIter cannot be less than 1.");
            }
        }

        m_numSeqsPerMB = m_numSeqsPerMBForAllEpochs[0];
        m_pMBLayout->Init(m_numSeqsPerMB, 0, true); // (SGD will ask before entering actual reading --TODO: This is hacky.)

        m_noData = false;

        string command(readerConfig("action", L"")); //look up in the config for the master command to determine whether we're writing output (inputs only) or training/evaluating (inputs and outputs)

        if (readerConfig.Exists("legacyMode"))
            RuntimeError("legacy mode has been deprecated\n");

        if (command == "write"){
            LogicError("Writer is not a reader!!!.");
        }
        else {
            source_ = std::shared_ptr<ISource>(new HTKMLFSource<ElemType>());
        }
    }

    //eldak: not clear why we templatize all over the place.
    template class NormalPacker<float>;
    template class NormalPacker<double>;
}}}