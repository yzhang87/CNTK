#include "stdafx.h"

#include "implementations.h"

void main()
{
    const size_t number_of_epochs = 3;
    const size_t minibatch_size = 50;

    config_parameters parameters;
    memory_provider_ptr provider(new memory_provider);

    epoch_configuration epoch_config;
    epoch_config.worker_rank = 0;
    epoch_config.number_of_workers = 1;
    epoch_config.total_size = 1066;

    reader_ptr reader = create_reader(parameters, provider);

    auto inputs = reader->get_inputs();

    for (size_t current_epoch = 0; current_epoch < number_of_epochs; ++current_epoch)
    {
        epoch_config.minibatch_size = minibatch_size;
        epoch_ptr epoch = reader->start_next_epoch(epoch_config);

        std::map<size_t, input_ptr> mb;
        while (epoch->read_minibatch(mb))
        {
            mb.clear();
        }
    }
}