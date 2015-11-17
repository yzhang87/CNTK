// Represents a timeline.

#pragma once

#include <string>
#include <vector>

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            typedef std::string sequenceId;

            class Timeline
            {
            public:
                std::vector<std::tuple<sequenceId, size_t>> timeline;
            };
        }
    }
}