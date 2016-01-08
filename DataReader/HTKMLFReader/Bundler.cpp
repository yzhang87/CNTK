#include "stdafx.h"
#include "Bundler.h"
#include "ConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

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
