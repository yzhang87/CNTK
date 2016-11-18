//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Function.h"
#include "InputAndParamNodes.h"

namespace CNTK
{
    Variable::Variable(const FunctionPtr& function)
        : Variable(function->Output())
    {
    }

    FunctionPtr Variable::Owner() const 
    {
        if (m_dataFields->m_ownerFunction != nullptr)
            return m_dataFields->m_ownerFunction->shared_from_this();
        else
            return nullptr;
    }

    Variable::operator FunctionPtr() const
    {
        auto varOwner = Owner();
        if (varOwner)
            return CompositeFunction::Create(varOwner, varOwner->Name());
        else
            return Combine({ *this });
    }

    NDArrayViewPtr Variable::Value() const
    {
        if (!IsConstant() && !IsParameter())
            LogicError("Only Variables of kind Parameter and Constant have a Value!");

        if (m_dataFields->m_value == nullptr)
        {
            assert(m_dataFields->m_valueInitializer);
            assert(m_dataFields->m_valueInitializationDevice);

            switch (GetDataType())
            {
            case DataType::Float:
            {
                m_dataFields->m_value = CreateValueFromParameterInitializer<float>(Shape(), *m_dataFields->m_valueInitializer, *m_dataFields->m_valueInitializationDevice);
                break;
            }
            case DataType::Double:
            {
                m_dataFields->m_value = CreateValueFromParameterInitializer<double>(Shape(), *m_dataFields->m_valueInitializer, *m_dataFields->m_valueInitializationDevice);
                break;
            }
            default:
                LogicError("Unsupported DataType %s", DataTypeName(GetDataType()));
                break;
            }

            m_dataFields->m_valueInitializer = nullptr;
            m_dataFields->m_valueInitializationDevice = nullptr;
        }

        assert(m_dataFields->m_value != nullptr);
        return m_dataFields->m_value;
    }

    static const std::wstring InitializerTypeAttributeName = L"initializerType";
    static const std::wstring OutputRankAttributeName = L"outputRank";
    static const std::wstring FilterRankAttributeName = L"filterRank";
    static const std::wstring ValueAttributeName = L"value";
    static const std::wstring ScaleAttributeName = L"scale";
    static const std::wstring RandomSeedAttributeName = L"randomSeed";
    static const std::wstring KernelWidthAttributeName = L"kernelWidth";
    static const std::wstring KernelHeightAttributeName = L"kernelHeight";

    void Variable::VariableFields::SetValueInitialization(const ParameterInitializer& initializationConfig, const DeviceDescriptor& device)
    {
        if (m_value != nullptr)
            LogicError("Value initialization config cannot be set if a value already exists");

        assert(!m_valueInitializer);
        assert(!m_valueInitializationDevice);

        if (initializationConfig.Contains(FilterRankAttributeName))
        {
            auto filterRank = (int)initializationConfig[FilterRankAttributeName].Value<size_t>();
            auto outputRank = (int)initializationConfig[OutputRankAttributeName].Value<size_t>();
            if ((filterRank + outputRank) > m_shape.Rank())
                InvalidArgument("Sum of filter rank (%d) and output rank (%d) of the parameter initializer cannot exceed the Parameter's rank(%d)", filterRank, outputRank, (int)m_shape.Rank());
        }

        m_valueInitializer.reset(new ParameterInitializer(initializationConfig));
        m_valueInitializationDevice.reset(new DeviceDescriptor(device));
    }

    namespace Internal
    {
        static std::atomic<unsigned long> s_fixedRandomSeed(0);
        void SetFixedRandomSeed(unsigned long fixedRandomSeed)
        {
            s_fixedRandomSeed.store(fixedRandomSeed);
        }
    }

    static std::atomic<unsigned long> s_currentRandomSeed(1);
    unsigned long DefaultRandomSeed()
    {
        auto currentFixedRandomSeed = Internal::s_fixedRandomSeed.load();
        if (currentFixedRandomSeed != 0)
            return currentFixedRandomSeed;

        return s_currentRandomSeed++;
    }

    static ParameterInitializer CreateInitializer(const std::wstring& initializerTypeName, int outputRank, int filterRank, double scale, unsigned long seed)
    {
        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = initializerTypeName;
        initConfig[OutputRankAttributeName] = (size_t)outputRank;
        initConfig[FilterRankAttributeName] = (size_t)filterRank;
        initConfig[ScaleAttributeName] = scale;
        initConfig[RandomSeedAttributeName] = (size_t)seed;

        return initConfig;
    }

    ParameterInitializer ConstantInitializer(double value)
    {
        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = Microsoft::MSR::CNTK::ConstantInitializerTypeName;
        initConfig[ValueAttributeName] = value;

        return initConfig;
    }

    ParameterInitializer UniformInitializer(double scale, unsigned long seed)
    {
        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = Microsoft::MSR::CNTK::UniformInitializerTypeName;
        initConfig[ScaleAttributeName] = scale;
        initConfig[RandomSeedAttributeName] = (size_t)seed;

        return initConfig;
    }

    ParameterInitializer GaussianInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::GaussianInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer XavierInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::XavierInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer GlorotUniformInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::GlorotUniformInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer GlorotNormalInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::GlorotNormalInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer HeUniformInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::HeUniformInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer HeNormalInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::HeNormalInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer BilinearInitializer(size_t kernelWidth, size_t kernelHeight)
    {
        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = Microsoft::MSR::CNTK::BilinearInitializerTypeName;
        initConfig[KernelWidthAttributeName] = kernelWidth;
        initConfig[KernelHeightAttributeName] = kernelHeight;

        return initConfig;
    }

    Variable::Variable(const NDShape& shape, VariableKind varType, CNTK::DataType dataType, Function* ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid)
        : m_dataFields(MakeSharedObject<VariableFields>(shape, varType, dataType, ownerFunction, value, needsGradient, dynamicAxes, isSparse, name, uid))
    {}

    template <typename ElementType>
    /*static*/ NDArrayViewPtr Variable::CreateValueFromParameterInitializer(const NDShape& shape, const ParameterInitializer& initConfig, const DeviceDescriptor& device)
    {
        auto dataType = AsDataType<ElementType>();
        auto value = MakeSharedObject<NDArrayView>(dataType, shape, device);
        auto valueMatrix = value->template GetWritableMatrix<ElementType>();
        auto initializerType = initConfig[InitializerTypeAttributeName].Value<std::wstring>();
        if (initializerType == Microsoft::MSR::CNTK::ConstantInitializerTypeName)
        {
            auto constantInitValue = initConfig[ValueAttributeName].Value<double>();
            valueMatrix->SetValue((ElementType)constantInitValue);
        }
        else if (initializerType == Microsoft::MSR::CNTK::BilinearInitializerTypeName)
        {
            auto kernelWidth = initConfig[KernelWidthAttributeName].Value<size_t>();
            auto kernelHeight = initConfig[KernelHeightAttributeName].Value<size_t>();

            Microsoft::MSR::CNTK::LearnableParameter<ElementType>::InitBilinear(*valueMatrix, AsTensorShape(shape), kernelWidth, kernelHeight, AsCNTKImplDeviceId(device));
        }
        else
        {
            auto randomSeed = (unsigned long)initConfig[RandomSeedAttributeName].Value<size_t>();
            auto scale = initConfig[ScaleAttributeName].Value<double>();
            int outputRank = DefaultParamInitOutputRank, filterRank = DefaultParamInitFilterRank;
            if (initializerType != Microsoft::MSR::CNTK::UniformInitializerTypeName)
            {
                outputRank = (int)initConfig[OutputRankAttributeName].Value<size_t>();
                filterRank = (int)initConfig[FilterRankAttributeName].Value<size_t>();
            }

            Microsoft::MSR::CNTK::LearnableParameter<ElementType>::InitRandom(*valueMatrix, AsTensorShape(shape), initializerType, randomSeed, (ElementType)scale, filterRank, outputRank, false, AsCNTKImplDeviceId(device));
        }

        return value;
    }
}