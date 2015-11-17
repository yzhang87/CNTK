#pragma once

#include "../interfaces/ISource.h"

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            class HTKMLFSource : public ISource
            {
            public:
                HTKMLFSource(/*IBlockReader[] readers, framemode, inputs, ... config*/)
                {}
            };
        }
    }
}