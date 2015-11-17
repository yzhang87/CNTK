#pragma once

#include <memory>
#include "../interfaces/IRandomizer.h"

class RollingWindowRandomizer : public IRandomizer
{
	int rank_;
	int numberOfWorkers_;
	int rollingWindowSize_;

public:
	RollingWindowRandomizer(std::shared_ptr<ISource> source, int rank, int numberOfWorkers, int rollingWindowSize)
		: IRandomizer(source)
		, rank_(rank)
		, numberOfWorkers_(numberOfWorkers)
		, rollingWindowSize_(rollingWindowSize)
	{}
};
