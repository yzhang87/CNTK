#pragma once

#include "reader_interface.h"
#include "inner_interfaces.h"

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

struct physical_timeline : timeline
{
    // Specific physical location per file format sequence
};

class sequence_reader
{};

class file_reader : public block_reader
{

    virtual void get(char* buffer, size_t offset, size_t size) override
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

};

typedef std::shared_ptr<block_reader> block_reader_ptr;
typedef std::shared_ptr<file_reader> file_reader_ptr;

class scp_reader
{
public:
    scp_reader(block_reader_ptr scp);
    physical_timeline get_timeline();
};

typedef std::shared_ptr<scp_reader> scp_reader_ptr;

class htk_sequence_reader : public sequence_reader
{
public:
    htk_sequence_reader(block_reader_ptr features, augmentation_descriptor, const physical_timeline& timeline);
};

typedef std::shared_ptr<htk_sequence_reader> htk_sequence_reader_ptr;

class mlf_sequence_reader : public sequence_reader
{
    mlf_sequence_reader(block_reader_ptr lables, block_reader_ptr states, const physical_timeline& timeline);
};

typedef std::shared_ptr<mlf_sequence_reader> mlf_sequence_reader_ptr;

class htkmlf_sequencer : sequencer
{
    htkmlf_sequencer(htk_sequence_reader_ptr, mlf_sequence_reader_ptr, scp_reader_ptr);

    virtual timeline& get_timeline() const override;
    virtual std::vector<input_description_ptr> get_inputs() const override;
    virtual std::map<size_t, sequence> get_sequence_by_id(size_t id) override;
};

typedef std::shared_ptr<htk_sequence_reader> htk_sequence_reader_ptr;

class packer {};

class chunk_randomizer : randomizer
{
public:
    chunk_randomizer(sequencer_ptr, size_t chunk_size, int seed);

    virtual std::vector<input_description_ptr> get_inputs() const override;
    virtual std::map<size_t, sequence> get_next_sequence() override;
};

class rolling_window_randomizer : randomizer
{};

class normal_packer : public reader
{
public:
    normal_packer(memory_provider_ptr, transformer_ptr, const config_parameters& config) {}

    virtual std::vector<input_description_ptr> get_inputs() override;
    virtual epoch_ptr start_next_epoch(const epoch_configuration& config) override;
};

class bptt_packer : packer
{};


reader_ptr create_reader(config_parameters& parameters, memory_provider_ptr memory_provider)
{
    // The code below will be split between the corresponding factory methods with appropriate
    // extraction of required parameters from the config.
    // Parameters will also be combined in the appropriate structures when needed.

    // Read parameters from config
    const int chunk_size = parameters["..."];
    const int seed = parameters["..."];

    // Read scp and form initial timeline
    block_reader_ptr scp(new file_reader(...));
    scp_reader_ptr t(new scp_reader(scp));


    // Create sequence readers to be combined by the sequencer.
    block_reader_ptr features_reader(new file_reader(...));
    htk_sequence_reader_ptr features(new htk_sequence_reader(features_reader, augmentation_descriptor(), t->get_timeline()));

    block_reader_ptr lables_reader(new file_reader(...));
    block_reader_ptr states_reader(new file_reader(...));
    mlf_sequence_reader_ptr labels(new mlf_sequence_reader(lables_reader, states_reader, t->get_timeline()));
    sequencer_ptr sequencer(new htkmlf_sequencer(features, labels, t));

    // Create randomizer and form randomized timeline.
    transformer_ptr randomizer(new chunk_randomizer(sequencer, chunk_size, seed));

    // Create the packer that will consume the sequences from the randomizer and will
    // pack them into efficient representation using the memory provider.
    return reader_ptr(new normal_packer(memory_provider, randomizer, parameters));
}
