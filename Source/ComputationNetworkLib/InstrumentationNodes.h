//
// <copyright file="InstrumentationNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "ConvolutionalNodes.h"
#include "Matrix.h"
#include "TensorView.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Prints out input matrices (on forward prop) and gradients (on backward prop)
    template<class ElemType>
    class TraceNode : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Trace"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(TraceNode);
        TraceNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        {
        }

        virtual void /*ComputationNode::*/BackpropTo(const size_t inputIndex, const FrameRange & fr) override
        {
            Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);
            Matrix<ElemType> sliceOutputGrad = GradientFor(fr);
            sliceOutputGrad.Print("<- Backprop <- ");
            sliceInput0Grad += sliceOutputGrad;
        }

        virtual bool OutputUsedInComputingInputNodesGradients() const override
        {
            // The TraceNode does not require its output value for computing
            // the gradients of its input nodes
            return false;
        }

        virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
        {
            // The TraceNode does not require any of it's input's values for computing
            // the gradients of its input nodes
            UNREFERENCED_PARAMETER(childIndex);
            return false;
        }

        virtual void UpdateFunctionMBSize() override
        {
            Base::UpdateFunctionMBSize();
        }

        virtual void /*ComputationNode::*/ForwardProp(const FrameRange & fr) override
        {
            Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
            sliceInput0Value.Print("-> ForwardProp ->");
            Matrix<ElemType> sliceOutputValue = ValueFor(fr);
            sliceOutputValue.SetValue(sliceInput0Value);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateUnaryMap(isFinalValidationPass);
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<TraceNode<ElemType>>(nodeP);
                node; // Do stuff
            }
        }
    private:
    };

    template class TraceNode<float>;
    template class TraceNode<double>;

} } }