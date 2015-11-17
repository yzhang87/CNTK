// Main interface of the reader. Provides a sequence of intputs to the network.

#pragma once

#include <memory>

#include "IMemoryProvider.h"
#include "ISequencer.h"
#include "ProcessingUnit.h"

class IPacker 
{
	std::shared_ptr<IMemoryProvider> memoryProvider_;
	std::shared_ptr<ISequencer> source_;
	size_t minibatchSize_;

public:
	IPacker(std::shared_ptr<IMemoryProvider> provider, size_t minibatchSize, std::shared_ptr<ISequencer> source/*, PackerConfig inputs*/)
		: memoryProvider_(provider)
		, source_(source)
		, minibatchSize_(minibatchSize)
	{}

    virtual std::shared_ptr<ProcessingUnit>* getNextProcessingUnit() = 0;
	virtual ~IPacker() = 0 {}
};
