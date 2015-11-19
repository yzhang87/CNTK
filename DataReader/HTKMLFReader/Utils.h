#pragma once

#include <string>
#include <vector>
#include <commandArgUtil.h>

namespace Microsoft {
    namespace MSR {
        namespace CNTK {


            // GetFileConfigNames - determine the names of the features and labels sections in the config file
            // features - [in,out] a vector of feature name strings
            // labels - [in,out] a vector of label name strings
            void GetDataNamesFromConfig(
                const Microsoft::MSR::CNTK::ConfigParameters& readerConfig,
                std::vector<std::wstring>& features,
                std::vector<std::wstring>& labels,
                std::vector<std::wstring>& hmms,
                std::vector<std::wstring>& lattices)
            {
                for (auto iter = readerConfig.begin(); iter != readerConfig.end(); ++iter)
                {
                    auto pair = *iter;
                    Microsoft::MSR::CNTK::ConfigParameters temp = iter->second;
                    // see if we have a config parameters that contains a "file" element, it's a sub key, use it
                    if (temp.ExistsCurrent("scpFile"))
                    {
                        features.push_back(msra::strfun::utf16(iter->first));
                    }
                    else if (temp.ExistsCurrent("mlfFile") || temp.ExistsCurrent("mlfFileList"))
                    {
                        labels.push_back(msra::strfun::utf16(iter->first));
                    }
                    else if (temp.ExistsCurrent("phoneFile"))
                    {
                        hmms.push_back(msra::strfun::utf16(iter->first));
                    }
                    else if (temp.ExistsCurrent("denlatTocFile"))
                    {
                        lattices.push_back(msra::strfun::utf16(iter->first));
                    }

                }
            }

            void ExpandDotDotDot(wstring & featPath, const wstring & scpPath, wstring & scpDirCached)
            {
                wstring delim = L"/\\";

                if (scpDirCached.empty())
                {
                    scpDirCached = scpPath;
                    wstring tail;
                    auto pos = scpDirCached.find_last_of(delim);
                    if (pos != wstring::npos)
                    {
                        tail = scpDirCached.substr(pos + 1);
                        scpDirCached.resize(pos);
                    }
                    if (tail.empty()) // nothing was split off: no dir given, 'dir' contains the filename
                        scpDirCached.swap(tail);
                }
                size_t pos = featPath.find(L"...");
                if (pos != featPath.npos)
                    featPath = featPath.substr(0, pos) + scpDirCached + featPath.substr(pos + 3);
            }
        }
    }
}