#pragma once

#include "reader_interface.h"

class epoch_impl : epoch
{
public:
    virtual std::map<input_id, input_ptr> read_minibatch();
    virtual ~epoch_impl() = 0 {};
};

class htkmlf_reader : reader
{
public:
    htkmlf_reader(const config_parameters& parameters, memory_provider_ptr memory_provider)
    {
    }

    virtual std::vector<input_description_ptr> get_inputs()
    {
        throw std::logic_error("not implemented");
    }

    virtual epoch_ptr start_next_epoch(const epoch_configuration& config)
    {
        throw std::logic_error("not implemented");
    }

    virtual ~htkmlf_reader(){};
};

reader_ptr create_reader(const config_parameters& parameters, memory_provider_ptr memory_provider)
{
    throw std::logic_error("not implemented");
}
