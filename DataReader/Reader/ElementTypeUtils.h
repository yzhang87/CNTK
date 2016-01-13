//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include <vector>
#include <memory>
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Returns the size of the type.
    inline size_t GetSizeByType(ElementType type)
    {
        switch (type)
        {
        case ElementType::tfloat:
            return sizeof(float);
        case ElementType::tdouble:
            return sizeof(double);
        case ElementType::tatom:
            return sizeof(char);
        default:
            RuntimeError("Unsupported type '%d'", type);
        }
    }
}}}
