#pragma once

#include "Packer.h"
#include "ISource.h"
#include "commandArgUtil.h" // for intargvector
#include "sequences.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    template<class ElemType>
    class NormalPacker : public Packer
    {
    public:
        NormalPacker(/*std::shared_ptr<IMemoryProvider> provider, size_t minibatchSize, std::shared_ptr<ISequencer> source,*/ const ConfigParameters inputs);

        virtual std::shared_ptr<ProcessingUnit>* getNextProcessingUnit() override
        {
            throw std::logic_error("The method or operation is not implemented.");
        }

    private:
        MBLayoutPtr m_pMBLayout;
        intargvector m_numSeqsPerMBForAllEpochs;
        size_t m_numSeqsPerMB;                  // requested number of parallel sequences
        bool m_noData;

        // eldak: For now we have it here, but should be injected probably from the factory.
        // move as soon as randomizer is in place.
        std::shared_ptr<ISource> source_;
    };
}}}
