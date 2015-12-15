#include "stdafx.h"
#include "MLFDataDeserializer.h"
#include "ConfigHelper.h"
#include "htkfeatio.h"
#include "msra_mgram.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    MLFDataDeserializer::MLFDataDeserializer(const ConfigParameters& label, size_t elementSize)
        : m_mlfPaths(std::move(ConfigHelper::GetMlfPaths(label)))
        , m_elementSize(elementSize)
    {
        ConfigHelper::CheckLabelType(label);

        m_dimension = ConfigHelper::GetLabelDimension(label);
        m_layout = std::make_shared<ImageLayout>(std::move(std::vector<size_t> { m_dimension }));

        m_stateListPath = label(L"labelMappingFile", L"");

        // TODO: currently assumes all Mlfs will have same root name (key)
        // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files

        // get labels
        const double htktimetoframe = 100000.0; // default is 10ms

        const msra::lm::CSymbolSet* wordTable = nullptr;
        std::map<string, size_t>* symbolTable = nullptr;

        msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence> labels
            (m_mlfPaths, std::set<wstring>(), m_stateListPath, wordTable, symbolTable, htktimetoframe);

        // Make sure 'msra::asr::htkmlfreader' type has a move constructor
        static_assert(
            std::is_move_constructible <
            msra::asr::htkmlfreader < msra::asr::htkmlfentry,
            msra::lattices::lattice::htkmlfwordsequence >> ::value,
            "Type 'msra::asr::htkmlfreader' should be move constructible!");

        size_t numClasses = 0; // TODO same as m_dimension?

        size_t numSequences = labels.size();
        m_sequences.reserve(numSequences);
        m_sequencesP.reserve(numSequences);

        MLFSequenceDescription description;
        description.id = 0;
        description.isValid = true; // right now we throw for invalid sequences
        // TODO .chunk, .key

        // Note: this is only checking that frames within a sequence are contiguous
        for (auto l : labels)
        {

            const auto & labseq = l.second;

            assert(0 < labseq.size()); // TODO

            description.key = l.first;
            description.numberOfSamples = labseq[0].firstframe;

            foreach_index(i, labseq)
            {
                // TODO Why will these yield a run-time error as opposed to making the utterance invalid?

                const auto & e = labseq[i];
                if ((i == 0 && e.firstframe != 0) ||
                    (i > 0 && labseq[i - 1].firstframe + labseq[i - 1].numframes != e.firstframe))
                {
                    RuntimeError("minibatchutterancesource: labels not in consecutive order MLF in label set: %ls", l.first.c_str());
                }

                if (e.classid >= m_dimension)
                {
                    RuntimeError("minibatchutterancesource: class id %llu exceeds model output dimension %llu in file %ls",
                        e.classid, m_dimension, l.first.c_str());
                }

                if (e.classid != static_cast<msra::dbn::CLASSIDTYPE>(e.classid))
                {
                    RuntimeError("CLASSIDTYPE has too few bits");
                }

                for (size_t t = e.firstframe; t < e.firstframe + e.numframes; t++)
                {
                    m_classIds.push_back(e.classid);
                }

                numClasses = max(numClasses, static_cast<size_t>(1u + e.classid));
            }

            // append a boundary marker marker for checking
            m_classIds.push_back(static_cast<msra::dbn::CLASSIDTYPE>(-1));

            m_sequences.push_back(description);
            m_sequencesP.push_back(&m_sequences[description.id]);

            description.id++;
            description.sequenceStart = m_classIds.size(); // TODO
        }
    }

    void Microsoft::MSR::CNTK::MLFDataDeserializer::SetEpochConfiguration(const EpochConfiguration& /*config*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    TimelineP Microsoft::MSR::CNTK::MLFDataDeserializer::GetSequenceDescriptions() const
    {
        return m_sequencesP;
    }

    Microsoft::MSR::CNTK::InputDescriptionPtr Microsoft::MSR::CNTK::MLFDataDeserializer::GetInput() const
    {
        InputDescriptionPtr input = std::make_shared<InputDescription>();
        input->id = 0;
        input->sampleLayout = std::make_shared<ImageLayout>(std::move(std::vector<size_t>{ m_dimension }));
        return input;
    }

    Microsoft::MSR::CNTK::Sequence Microsoft::MSR::CNTK::MLFDataDeserializer::GetSequenceById(size_t /*id*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    Sequence MLFDataDeserializer::GetSampleById(size_t sequenceId, size_t sampleId)
    {
        size_t label = m_classIds[m_sequences[sequenceId].sequenceStart + sampleId];

        Sequence r;
        if (m_elementSize == sizeof(float))
        {
            float* tmp = new float[m_dimension];
            memset(tmp, 0, m_elementSize * m_dimension);
            tmp[label] = 1;
            r.data = tmp;
            r.numberOfSamples = 1;
        }
        else
        {
            double* tmp = new double[m_dimension];
            tmp[label] = 1;
            r.data = tmp;
            r.numberOfSamples = 1;
        }

        return r;
    }

    bool Microsoft::MSR::CNTK::MLFDataDeserializer::RequireChunk(size_t /*chunkIndex*/)
    {
        assert(false);
        throw std::logic_error("The method or operation is not implemented.");
    }

    void Microsoft::MSR::CNTK::MLFDataDeserializer::ReleaseChunk(size_t /*chunkIndex*/)
    {
        assert(false);
        throw std::logic_error("The method or operation is not implemented.");
    }

}}}
