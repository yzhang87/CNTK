// Main interface that any file/network/stream should provide in order to provide raw data to a CNTK reader.
// It wraps simple IO operations and can be reused in many readers.

#pragma once

class IBlockReader 
{
public:
    virtual char* get(size_t offset, size_t size) = 0;
	virtual ~IBlockReader() {}
};