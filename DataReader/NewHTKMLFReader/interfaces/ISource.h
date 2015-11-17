// Represents a source - entity that drives the process of generating sequences.
// Responsible for synchronous reading from different inputs and providing sequences to the transformations.
// Responsible for generation of initial timeline based on the available training set.

#pragma once

#include "Timeline.h"

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            class ISource
            {
            public:
                virtual Timeline& getTimeline() = 0;
                virtual std::map<std::string, std::vector<sequence>> getSequenceById(std::vector<sequenceId> ids) = 0;
                virtual ~ISource() = 0 {}
            };
        }
    }
}