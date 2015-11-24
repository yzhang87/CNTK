#pragma once

#include <vector>
#include <memory>
#include <map>

class config_parameters
{
};

// Epoch configuration.
struct epoch_configuration
{
    size_t worker_rank;
    size_t number_of_workers;

    size_t minibatch_size;
    size_t total_size;

    size_t number_of_sequences;
};

typedef size_t input_id;

// Input description.
struct input_description
{
    std::string name;
    input_id id;
    std::string target_layout_type;
    std::map<std::string, std::string> properties;
};
typedef std::shared_ptr<input_description> input_description_ptr;

class minibatch_layout
{
};

class tensor_layout
{
};

class layout
{
    minibatch_layout columns;
    tensor_layout rows;
};

typedef std::shared_ptr<layout> layout_ptr;

// Input data.
class input
{
    const char* get_data() const;
    const size_t get_data_size() const;
    const layout_ptr get_layout() const;
};
typedef std::shared_ptr<input> input_ptr;

// Memory provider. Should be used for allocating storage according to the layout.
class memory_provider
{
public:
    void* alloc(size_t element, size_t number_of_elements);
    void free(void* ptr);
};
typedef std::shared_ptr<memory_provider> memory_provider_ptr;

// Represents a single epoch.
class epoch
{
public:
    virtual bool read_minibatch(std::map<input_id, input_ptr> minibatch);
    virtual ~epoch() = 0 {};
};
typedef std::unique_ptr<epoch> epoch_ptr;

// Main reader interface. The border interface between the CNTK and reader.
class reader
{
public:
    virtual std::vector<input_description_ptr> get_inputs() = 0;
    virtual epoch_ptr start_next_epoch(const epoch_configuration& config) = 0;
    virtual ~reader() = 0 {};
};
typedef std::unique_ptr<reader> reader_ptr;


// Factory function for creating a reader.
reader_ptr create_reader(const config_parameters& parameters, memory_provider_ptr memory_provider);
