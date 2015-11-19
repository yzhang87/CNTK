// Implements a transformation of a sequence, where a sequence is a vector of frames.

#pragma once

#include <map>
#include <vector>
#include "sequence.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    class ISequencer
    {
    public:
        virtual std::map<std::string, std::vector<Sequence>> getNextSequences(size_t numberOfSequences) = 0;
        virtual ~ISequencer() = 0 {}
    };
}}}