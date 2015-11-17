// Represents a data per input for the processing unit.

#pragma once

class Metadata
{};

class Data
{
public:
    char* data;
    size_t size;
    Metadata layout;
};
