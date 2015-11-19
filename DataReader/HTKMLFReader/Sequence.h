// Represents a sequence, which is a vector of frames.

#pragma once

#include <vector>
#include "frame.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    class sequence
    {
        std::vector<Frame> frames;
    };
}}}