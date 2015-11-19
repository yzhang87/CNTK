#pragma once

#include "IRandomizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    class BlockRandomizer : public IRandomizer
    {
        int rank_;
        int numberOfWorkers_;

    public:
        BlockRandomizer(std::shared_ptr<ISource> source, int rank, int numberOfWorkers)
            : IRandomizer(source)
            , rank_(rank)
            , numberOfWorkers_(numberOfWorkers)
        {} // "chunk size" is hardcoded for now (cf. utterancesource.h)
    };
}}}