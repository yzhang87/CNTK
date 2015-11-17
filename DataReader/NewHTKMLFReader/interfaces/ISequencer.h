// Implements a transformation of a sequence, where a sequence is a vector of frames.

#pragma once

#include <map>
#include <vector>
#include "sequence.h"

class ISequencer
{
public:
    virtual std::map<std::string, std::vector<sequence>> getNextSequences(size_t numberOfSequences) = 0;
    virtual ~ISequencer() = 0 {}
};
