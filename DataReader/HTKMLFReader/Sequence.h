// Represents a sequence, which is a vector of frames.

#pragma once

#include <vector>
#include "frame.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    class Sequence
    {
    public:
        std::vector<Frame> frames;
    };
}}}