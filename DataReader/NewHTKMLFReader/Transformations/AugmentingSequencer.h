#pragma once

#include <memory>
#include "../interfaces/ISequencer.h"

class AugmentingSequencer : public ISequencer
{
	std::shared_ptr<ISequencer> source_;
	int contextLeft_;
	int contextRight_;

public:
	AugmentingSequencer(std::shared_ptr<ISequencer> source, int contextLeft, int contextRight)
		: source_(source)
		, contextLeft_(contextLeft)
		, contextRight_(contextRight)
	{}
};

