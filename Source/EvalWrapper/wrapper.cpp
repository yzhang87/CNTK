#include <windows.h>
#include <vcclr.h>
#include <string>
#include <utility>
#include <msclr\marshal_cppstd.h>

#include "Eval.h"

#using <System.dll>
#using <System.Collections.dll>

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Collections;

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK 
{
    template<typename ElemType>
    using GetEvalProc = void(*)(IEvaluateModel<ElemType>**);

    template<typename ElemType>
    public ref class IEvaluateModelManaged
    {
    public:
        IEvaluateModelManaged(String^ funcName)
        {
            /*
            pin_ptr<const WCHAR> dllname = PtrToStringChars("evaldll.dll");
            auto hModule = LoadLibrary(dllname);

            msclr::interop::marshal_context context;
            const std::string func = context.marshal_as<std::string>(funcName);
            auto procAddress = GetProcAddress(hModule, func.c_str());
            //auto getEvalProc = (GetEvalProc<ElemType>)procAddress;
            //getEvalProc(p_eval);
            */

            pin_ptr <IEvaluateModel<ElemType>*> p_eval = &m_eval;
            GetEvalF(p_eval);
        }

        void Init(String^ config)
        {
            msclr::interop::marshal_context context;
            const std::string stdConfig = context.marshal_as<std::string>(config);

            m_eval->Init(stdConfig);
        }

        void Destroy()
        {
            m_eval->Destroy();
        }

        void LoadModel(String^ modelFileName)
        {
            pin_ptr<const WCHAR> stdModelPath = PtrToStringChars(modelFileName);
            m_eval->LoadModel(stdModelPath);
        }

        void Evaluate(Dictionary<String^, List<ElemType>^>^ inputs, Dictionary<String^, List<ElemType>^>^ outputs)
        {
            std::map<std::wstring, std::vector<ElemType>*> stdInputs;
            std::map<std::wstring, std::vector<ElemType>*> stdOutputs;

            for each (auto item in inputs)
            {
                pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
                auto stdInput = new std::pair<std::wstring, std::vector<ElemType>*>(key, CopyList(item.Value->ToArray()));
                stdInputs.insert(*stdInput);
            }

            for each (auto item in outputs)
            {
                pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
                auto stdOutput = new std::pair<std::wstring, std::vector<ElemType>*>(key, CopyList(item.Value->ToArray()));
                stdOutputs.insert(*stdOutput);
            }

            //m_eval->StartEvaluateMinibatchLoop(L"HLast");
            m_eval->Evaluate(stdInputs, stdOutputs);

            std::vector<ElemType>* vector = stdOutputs.begin()->second;
            std::vector<ElemType> refVector = (*vector);
            int index = 0;
            auto enumerator = outputs->Keys->GetEnumerator();
            enumerator.MoveNext();
            String^ key = enumerator.Current;

            for (std::vector<ElemType>::iterator ii = refVector.begin(), e = refVector.end(); ii != e; ii++)
            {
                outputs[key][index++] = *ii;
            }
        }


    private:
        IEvaluateModel<ElemType> *m_eval;

        std::vector<ElemType>* CopyList(array<ElemType>^ list)
        {
            std::vector<ElemType>* lower = new std::vector<ElemType>();
            for each (auto item in list)
            {
                lower->push_back(item);
            }

            return lower;
        }
    };

    public ref class IEvaluateModelManagedF : IEvaluateModelManaged<float>
    {
    public:
        IEvaluateModelManagedF::IEvaluateModelManagedF()
            : IEvaluateModelManaged("GetEvalF")
        {
        }
    };
    /*
    public ref class IEvaluateModelManagedD : IEvaluateModelManaged<double>
    {
    public:
        IEvaluateModelManagedD::IEvaluateModelManagedD()
            : IEvaluateModelManaged("GetEvalD")
        {
        }
    };
    */
    void emit()
    {
        // This method tricks the compiler into emitting the methods of the classes
        // Refer to https://msdn.microsoft.com/en-us/library/ms177213.aspx for an
        // explanation to this insanity
        IEvaluateModelManagedF f;
        f.Init("");
        f.Evaluate(nullptr, nullptr);
        f.LoadModel("");
        f.Destroy();
        
        /*
        IEvaluateModelManagedD d;
        d.Init("");
        d.Evaluate(nullptr, nullptr);
        d.LoadModel("");
        d.Destroy();
        */
    }


}}}