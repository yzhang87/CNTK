// All randomizers have to implment the following interface.

#pragma once

#include "ISequencer.h"
#include "ISource.h"
#include <memory>

class IRandomizer : ISequencer 
{
	std::shared_ptr<ISource> source_;

public:
	IRandomizer(std::shared_ptr<ISource> source)
		: source_(source)
	{}

	virtual ~IRandomizer() = 0 {}
};
