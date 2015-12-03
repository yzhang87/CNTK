#include "stdafx.h"
#include "ScpDataDeserializer.h"
#include "BlockRandomizer.h"

using namespace msra::dbn;

namespace Microsoft { namespace MSR { namespace CNTK {

    std::vector<utterancedesc> ScpDataDeserializer::Parse(const std::vector<std::wstring>& featureScpFiles)
    {
        std::vector<utterancedesc> utterances;

        for (const auto& file : featureScpFiles)
        {
            utterancedesc u(msra::asr::htkfeatreader::parsedpath(file), 0);
            utterances.push_back(u);
        }

        return utterances;
    }

}}}