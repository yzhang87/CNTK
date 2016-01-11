//
// <copyright file="Bundler.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "DataDeserializer.h"
#include "commandArgUtil.h" // for ConfigParameters

namespace Microsoft { namespace MSR { namespace CNTK {

    class Bundler: public DataDeserializer
    {
        void operator=(const Bundler& other); // non-assignable

        bool m_framemode; // true -> actually return frame-level randomized frames (not possible in lattice mode)
        int m_verbosity;

        size_t m_totalframes; // total frames (same as classids.size() if we have labels)
        size_t m_chunksinram; // (for diagnostics messages)

        std::vector<InputDescriptionPtr> m_inputs;
        std::vector<DataDeserializerPtr> m_deserializers;
        DataDeserializerPtr m_driver;

    public:
        Bundler(const ConfigParameters& readerConfig, bool framemode, int verbosity,
            DataDeserializerPtr driver, std::vector<DataDeserializerPtr> deserializers);

        virtual void SetEpochConfiguration(const EpochConfiguration& config) override;

        virtual const Timeline& GetSequenceDescriptions() const override;
        virtual std::vector<InputDescriptionPtr> GetInputs() const override;
        virtual std::vector<std::vector<Sequence>> GetSequencesById(const std::vector<size_t> & ids) override;
        virtual bool RequireChunk(size_t chunkindex) override;
        virtual void ReleaseChunk(size_t chunkIndex) override;
    };

    std::vector<DataDeserializerPtr> CreateDeserializers(const ConfigParameters& readerConfig,
        bool framemode,
        size_t elementSize);
}}}
