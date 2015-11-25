#include "stdafx.h"

#include "implementations.h"

void main()
{
    const size_t numberOfEpochs = 3;
    const size_t minibatchSize = 50;

    ConfigParameters parameters;
    MemoryProviderPtr provider(new MemoryProvider);

    EpochConfiguration epochConfig;
    epochConfig.workerRank = 0;
    epochConfig.numberOfWorkers = 1;
    epochConfig.totalSize = 1066;

    ReaderPtr reader = createReader(parameters, provider);

    auto inputs = reader->getInputs();

    for (size_t current_epoch = 0; current_epoch < numberOfEpochs; ++current_epoch)
    {
        epochConfig.minibatchSize = minibatchSize;
        EpochPtr epoch = reader->startNextEpoch(epochConfig);

        Minibatch mb;
        while (mb = epoch->readMinibatch())
        {
            // TODO
        }
    }
}
