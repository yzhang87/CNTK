#include "stdafx.h"
#include "MLFDataDeserializer.h"
#include "ConfigHelper.h"
#include "htkfeatio.h"
#include "msra_mgram.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    MLFDataDeserializer::MLFDataDeserializer(const ConfigParameters& label)
        : m_mlfPaths(std::move(ConfigHelper::GetMlfPaths(label)))
    {
        ConfigHelper::CheckLabelType(label);

        m_dimension = ConfigHelper::GetLabelDimension(label);
        m_layout = std::make_shared<ImageLayout>(std::move(std::vector<size_t> { m_dimension }));

        m_stateListPath = label(L"labelMappingFile", L"");

        // TODO: currently assumes all Mlfs will have same root name (key)
        // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files

        // get labels
        double htktimetoframe = 100000.0; // default is 10ms

        const msra::lm::CSymbolSet* wordTable = nullptr;
        std::map<string, size_t>* symbolTable = nullptr;

        msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence> labels
            (m_mlfPaths, std::set<wstring>(), m_stateListPath, wordTable, symbolTable, htktimetoframe);

        // Make sure 'msra::asr::htkmlfreader' type has a move constructor
         static_assert(
             std::is_move_constructible<
                msra::asr::htkmlfreader<msra::asr::htkmlfentry,
                msra::lattices::lattice::htkmlfwordsequence>>::value,
             "Type 'msra::asr::htkmlfreader' should be move constructible!");
    }

    void Microsoft::MSR::CNTK::MLFDataDeserializer::SetEpochConfiguration(const EpochConfiguration& /*config*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    TimelineP Microsoft::MSR::CNTK::MLFDataDeserializer::GetSequenceDescriptions() const
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    Microsoft::MSR::CNTK::InputDescriptionPtr Microsoft::MSR::CNTK::MLFDataDeserializer::GetInput() const
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    Microsoft::MSR::CNTK::Sequence Microsoft::MSR::CNTK::MLFDataDeserializer::GetSequenceById(size_t /*id*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    Microsoft::MSR::CNTK::Sequence Microsoft::MSR::CNTK::MLFDataDeserializer::GetSampleById(size_t /*sequenceId*/, size_t /*sampleId*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    bool Microsoft::MSR::CNTK::MLFDataDeserializer::RequireChunk(size_t /*chunkIndex*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    void Microsoft::MSR::CNTK::MLFDataDeserializer::ReleaseChunk(size_t /*chunkIndex*/)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }
}}}