//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "Reader.h"
#include "ImageTransformers.h"
#include "FrameModePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Imlementation of the image reader.
    class ImageReader : public Reader
    {
    public:
        ImageReader(MemoryProviderPtr provider,
            const ConfigParameters& parameters);

        // Description of streams that this reader provides.
        std::vector<StreamDescriptionPtr> GetStreams() override;

        // Starts a new epoch with the provided configuration.
        void StartEpoch(const EpochConfiguration& config) override;

        // Reads a single minibatch.
        Minibatch ReadMinibatch() override;

    private:
        // All streams this reader provides.
        std::vector<StreamDescriptionPtr> m_streams;

        // A head transformer in a list of transformers.
        TransformerPtr m_transformer;

        // Packer.
        FrameModePackerPtr m_packer;

        // Seed for the random generator.
        unsigned int m_seed;

        // Memory provider (TODO: this will possibly change in the near future.)
        MemoryProviderPtr m_provider;
    };
}}}
