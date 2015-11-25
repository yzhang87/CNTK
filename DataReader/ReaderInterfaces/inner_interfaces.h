#pragma once

#include <vector>
#include "reader_interface.h"

// Defines the encoding of a frame.
struct frame_description
{
    std::vector<size_t> frame_dimensions;
    size_t element_size;
};

// Defines identifier and length of a sequence.
struct sequence_description
{
    size_t id;
    size_t length;
};

// Defines a sequences, which consists of sequences description and a number
// of frames, which have the same encoding and are layed out in memory contiguously.
struct sequence
{
    sequence_description* description;
    frame_description* frame_description;
    char* data;
    size_t number_of_frames;
};

// Low-level input interface (for file, network, etc.).
// Memory buffers to fill data into are provided by the caller.
class block_reader
{
public:
    virtual void get(char* buffer, size_t offset, size_t size) = 0;
    virtual ~block_reader() = 0 {}
};

// Timeline specifies a vector of sequence IDs and lengths.
// This information is exposed by a sequencer, e.g., to be used by the randomizer.
typedef std::vector<sequence_description> timeline;

// Timeline offsets specify file offset of sequences. These are used internally
// of a sequence reader or a sequencer.
typedef std::vector<size_t> timeline_offsets;

// A sequencer composes timeline information and a number of sequence readers, providing
// random-access to the timeline as well as the composed sequence readers.
class sequencer
{
public:
    virtual timeline& get_timeline() const = 0;
    virtual std::vector<input_description_ptr> get_inputs() const = 0;
    virtual std::map<input_id, sequence> get_sequence_by_id(size_t id) = 0;
    virtual ~sequencer() = 0 {};
};

typedef std::shared_ptr<sequencer> sequencer_ptr;

// Defines context augmentation (to the left and to the right).
// This will be specified as a construction parameter to sequence reader.
struct augmentation_descriptor
{
    size_t context_left;
    size_t context_right;
};

// Provides input descriptions and sequential access to sequences.
class transformer
{
public:
    virtual std::vector<input_description_ptr> get_inputs() const = 0;
    virtual ~transformer() = 0 {}
    virtual std::map<input_id, sequence> get_next_sequence() = 0;
};

typedef std::shared_ptr<transformer> transformer_ptr;

// A randomizer implements sequence randomization for a sequencer and
// additional parameters given at construction time.
// Note: chunk-level randomization can be implemented based on sequence lengths
// exposed through the sequencer's timeline method.
class randomizer : public transformer
{
};

class image_cropper : public transformer
{
};
