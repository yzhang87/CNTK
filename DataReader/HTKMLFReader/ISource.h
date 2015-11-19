// Represents a source - entity that drives the process of generating sequences.
// Responsible for synchronous reading from different inputs and providing sequences to the transformations.
// Responsible for generation of initial timeline based on the available training set.

#pragma once

#include <vector>
#include <map>
#include "Timeline.h"
#include "Sequence.h"

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            struct InputDefinition
            {
                std::wstring name;
                size_t id;
                std::vector<size_t> dimensions;
                size_t elementSize;
            };

            class ISource
            {
            public:
                virtual Timeline& getTimeline() = 0;
                virtual std::vector<InputDefinition> getInputs() = 0;
                virtual std::map<size_t, Sequence> getSequenceById(sequenceId id) = 0;
                virtual ~ISource() {}
            };
        }
    }
}