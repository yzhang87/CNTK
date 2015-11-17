// Represents a single frame that can be either an array of floats or doubles.

#pragma once

#include <vector>

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            class frame
            {
            public:
                std::vector<char> features;
            };
        }
    }
}