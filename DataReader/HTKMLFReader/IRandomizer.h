// All randomizers have to implement the following interface.

#pragma once

#include "ISequencer.h"
#include "ISource.h"
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {
    class IRandomizer : ISequencer
    {
        std::shared_ptr<ISource> source_;

    public:
        IRandomizer(std::shared_ptr<ISource> source)
            : source_(source)
        {}

        virtual ~IRandomizer() = 0 {}
    };
}}}