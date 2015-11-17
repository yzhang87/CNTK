// Represents a processing unit for all the inputs.
#include "Data.h"

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            class ProcessingUnit
            {
            public:
                std::map<std::string, std::tuple<Data>> data;
            };
        }
    }
}