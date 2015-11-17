// Represents a data per input for the processing unit.

#pragma once

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
        }
    }
}