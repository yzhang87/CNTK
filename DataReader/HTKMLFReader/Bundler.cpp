#include "stdafx.h"
#include "Bundler.h"
#include <DataReader.h>
#include "Utils.h"
#include "ConfigHelper.h"
#include "msra_mgram.h"
#include <DataTensor.h>
#include "HTKDataDeserializer.h"
#include "MLFDataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    std::vector<DataDeserializerPtr> CreateDeserializers(const ConfigParameters& readerConfig,
        bool framemode,
        size_t elementSize)
    {
        std::vector<std::wstring> featureNames;
        std::vector<std::wstring> labelNames;

        std::vector<std::wstring> notused;
        Utils::GetDataNamesFromConfig(readerConfig, featureNames, labelNames, notused, notused);
        if (featureNames.size() < 1 || labelNames.size() < 1)
        {
            // eldak: Don't we support unsupervised training?
            InvalidArgument("network needs at least 1 input and 1 output specified!");
        }

        std::vector<HTKDataDeserializerPtr> featureDeserializers;
        std::vector<MLFDataDeserializerPtr> labelDeserializers;

        std::vector<InputDescriptionPtr> inputs;
        for (const auto& featureName : featureNames)
        {
            auto deserializer = std::make_shared<HTKDataDeserializer>(readerConfig(featureName), elementSize, framemode, featureName);
            featureDeserializers.push_back(deserializer);
        }

        assert(featureDeserializers.size() == 1);

        for (const auto& labelName : labelNames)
        {
            auto deserializer = std::make_shared<MLFDataDeserializer>(readerConfig(labelName), elementSize,
                featureDeserializers[0].get(), framemode, labelName);

            labelDeserializers.push_back(deserializer);
        }

        assert(labelDeserializers.size() == 1);

        std::vector<DataDeserializerPtr> deserializers;
        deserializers.insert(deserializers.end(), featureDeserializers.begin(), featureDeserializers.end());
        deserializers.insert(deserializers.end(), labelDeserializers.begin(), labelDeserializers.end());

        // Checking end sequences.
        size_t expected = deserializers[0]->GetSequenceDescriptions().size();
        std::vector<bool> isValid(expected, true);
        for (auto d : deserializers)
        {
            const auto& sequences = d->GetSequenceDescriptions();
            if (sequences.size() != expected)
            {
                RuntimeError("We have some invalid alignment\n");
            }

            foreach_index(i, sequences)
            {
                isValid[i] = isValid[i] && sequences[i]->isValid;
                assert(isValid[i]);
            }
        }

        // shouldn't this be checked (again) later? more utterances can be invalidated...
        size_t invalidUtts = 0;
        foreach_index(i, isValid)
        {
            if (!isValid[i])
            {
                invalidUtts++;
            }
        }
        assert(invalidUtts == 0); // For us it's zero

        if (invalidUtts > isValid.size() / 2)
        {
            RuntimeError("minibatchutterancesource: too many files with inconsistent durations, assuming broken configuration\n");
        }
        else if (invalidUtts > 0)
        {
            fprintf(stderr, "Found inconsistent durations across feature streams in %llu out of %llu files\n", invalidUtts, isValid.size());
        }

        return deserializers;
    }

    Bundler::Bundler(
        const ConfigParameters& readerConfig,
        bool framemode,
        int verbosity,
        DataDeserializerPtr driver,
        std::vector<DataDeserializerPtr> deserializers)
        : m_deserializers(deserializers)
        , m_driver(driver)
    {
        m_framemode = framemode;
        m_chunksinram = 0;
        m_verbosity = readerConfig(L"verbosity", 2);
        m_verbosity = verbosity; // not needed

        std::vector<InputDescriptionPtr> inputs;
        for (auto d : deserializers)
        {
            for (auto i : d->GetInputs())
            {
                InputDescriptionPtr input = std::make_shared<InputDescription>();
                input->id = inputs.size();
                input->name = i->name;
                input->sampleLayout = i->sampleLayout;
                inputs.push_back(input);
            }
        }

        m_inputs = inputs;
    }

    bool Bundler::RequireChunk(size_t chunkindex)
    {
        // currently simply redirect
        // todo: we should have a mapping per deserializer actually.
        bool result = false;
        for (const auto& d: m_deserializers)
        {
            result |= d->RequireChunk(chunkindex);
        }

        return result;
    }

    void Bundler::ReleaseChunk(size_t chunkIndex)
    {
        // currently simply redirect
        // todo: we should have a mapping per deserializer actually.
        for (const auto& d : m_deserializers)
        {
            d->ReleaseChunk(chunkIndex);
        }
    }

    const TimelineP& Bundler::GetSequenceDescriptions() const
    {
        // TODO: we probably will take different deserializers from here.
        return m_driver->GetSequenceDescriptions();
    }

    std::vector<InputDescriptionPtr> Bundler::GetInputs() const
    {
        return m_inputs;
    }

    std::vector<Sequence> Bundler::GetSequenceById(size_t id)
    {
        std::vector<Sequence> result;
        for (auto& d : m_deserializers)
        {
            auto r = d->GetSequenceById(id);
            result.insert(result.end(), r.begin(), r.end());
        }
        return result;
    }

    void Bundler::SetEpochConfiguration(const EpochConfiguration& /*config*/)
    {
        // TODO do we keep SetEpochConfiguration(), now empty?
    }
}}}
