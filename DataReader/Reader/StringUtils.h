//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include <string>
#include <locale>

namespace Microsoft { namespace MSR { namespace CNTK {

    // Compares two strings ignoring the case.
    // TODO: Should be moved to common CNTK library.
    inline bool AreEqualIgnoreCase(const std::string& s1, const std::string& s2)
    {
        return std::equal(s1.begin(), s1.end(), s2.begin(), [](const char& a, const char& b) { return std::tolower(a) == std::tolower(b); });
    }

}}}
