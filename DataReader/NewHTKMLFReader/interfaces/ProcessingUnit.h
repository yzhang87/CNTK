// Represents a processing unit for all the inputs.

#pragma once

#include <memory>

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            class Metadata
            {};

            class Data
            {
            public:
                char* data;
                size_t size;
                Metadata layout;
            };

            class ProcessingUnit
            {
            public:
                std::map<std::string, std::shared_ptr<Data>> data;
            };
        }
    }
}