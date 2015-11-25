#pragma once

#include "reader_interface.h"
#include "inner_interfaces.h"

#include <string>

class EpochImplementation : public Epoch
{
public:
    virtual Minibatch readMinibatch()
    {
        return Minibatch();
    };
    virtual ~EpochImplementation() {};
};


class HtkmlfReader : public Reader
{
public:
    HtkmlfReader(const ConfigParameters& parameters, MemoryProviderPtr memoryProvider)
    {
    }

    virtual std::vector<InputDescriptionPtr> getInputs()
    {
        throw std::logic_error("not implemented");
    }

    virtual EpochPtr startNextEpoch(const EpochConfiguration& config)
    {
        throw std::logic_error("not implemented");
    }

    virtual ~HtkmlfReader(){};
};

struct PhysicalTimeline : Timeline
{
    // Specific physical location per file format Sequence
};

class SequenceReader
{};

class FileReader : public BlockReader
{
public:
    FileReader(std::string fileName);
    virtual ~FileReader() override
    {
    }

    virtual void get(char* buffer, size_t offset, size_t size) override
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

};

typedef std::shared_ptr<BlockReader> BlockReaderPtr;
typedef std::shared_ptr<FileReader> FileReaderPtr;

class ScpReader
{
public:
    ScpReader(BlockReaderPtr scp);
    PhysicalTimeline getTimeline();
};

typedef std::shared_ptr<ScpReader> ScpReaderPtr;

class HtkSequenceReader : public SequenceReader
{
public:
    HtkSequenceReader(BlockReaderPtr features, AugmentationDescriptor augmentationDescriptor, const PhysicalTimeline& timeline);
};

typedef std::shared_ptr<HtkSequenceReader> HtkSequenceReaderPtr;

class MlfSequenceReader : public SequenceReader
{
public:
    MlfSequenceReader(BlockReaderPtr lables, BlockReaderPtr states, const PhysicalTimeline& timeline);
};

typedef std::shared_ptr<MlfSequenceReader> MlfSequenceReaderPtr;

class HtkMlfSequencer : public Sequencer
{
public:
    HtkMlfSequencer(HtkSequenceReaderPtr, MlfSequenceReaderPtr, ScpReaderPtr);

    virtual Timeline& getTimeline() const override;
    virtual std::vector<InputDescriptionPtr> getInputs() const override;
    virtual std::map<size_t, Sequence> getSequenceById(size_t id) override;
};

typedef std::shared_ptr<HtkSequenceReader> HtkSequenceReaderPtr;

class ChunkRandomizer : public Randomizer
{
public:
    ChunkRandomizer(SequencerPtr, size_t chunkSize, int seed);

    virtual std::vector<InputDescriptionPtr> getInputs() const override;
    virtual std::map<size_t, Sequence> getNextSequence() override;
};

class RollingWindowRandomizer : public Randomizer
{
};

class NormalPacker : public Packer
{
public:
    NormalPacker(MemoryProviderPtr memoryProvider, TransformerPtr transformer, const ConfigParameters& config) {}

    virtual std::vector<InputDescriptionPtr> getInputs() override;
    virtual EpochPtr startNextEpoch(const EpochConfiguration& config) override;
};

class BpttPacker : public Packer
{
};

class ReaderFacade : public Reader
{
public:
    ReaderFacade(PackerPtr packer) {}
    virtual std::vector<InputDescriptionPtr> getInputs()
    {
        std::vector<InputDescriptionPtr> result;
        return result;
    };
    virtual EpochPtr startNextEpoch(const EpochConfiguration& config)
    {
        return std::make_unique<EpochImplementation>();
    };
    virtual ~ReaderFacade() { }
};

ReaderPtr createReader(ConfigParameters& parameters, MemoryProviderPtr memoryProvider)
{
    // The code below will be split between the corresponding factory methods with appropriate
    // extraction of required parameters from the config.
    // Parameters will also be combined in the appropriate structures when needed.

    // Read parameters from config
    const int chunkSize = std::stoi(parameters["..."]);
    const int seed = std::stoi(parameters["..."]);

    // Read scp and form initial Timeline
    auto scpFilename = parameters["scpFilename"];
    BlockReaderPtr scp(new FileReader(scpFilename));
    ScpReaderPtr t(new ScpReader(scp));

    // Create Sequence readers to be combined by the Sequencer.
    auto featureFilename = parameters["featureFilename"];
    BlockReaderPtr featureReader(new FileReader(featureFilename));
    HtkSequenceReaderPtr feature(new HtkSequenceReader(featureReader, AugmentationDescriptor(), t->getTimeline()));

    auto labelsFilename = parameters["labelsFilename"];
    BlockReaderPtr labelReader(new FileReader(labelsFilename));
    auto statesFilename = parameters["statesFilename"];
    BlockReaderPtr statesReader(new FileReader(statesFilename));
    MlfSequenceReaderPtr labels(new MlfSequenceReader(labelReader, statesReader, t->getTimeline()));
    SequencerPtr sequencer(new HtkMlfSequencer(feature, labels, t));

    // Create Randomizer and form randomized Timeline.
    TransformerPtr randomizer(new ChunkRandomizer(sequencer, chunkSize, seed));

    // Create the Packer that will consume the sequences from the Randomizer and will
    // pack them into efficient representation using the memory provider.
    PackerPtr packer = PackerPtr(new NormalPacker(memoryProvider, randomizer, parameters));

    return std::make_unique<ReaderFacade>(std::move(packer));
}
