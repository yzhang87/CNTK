#include <windows.h>
#include <vcclr.h>
#include <string>
#include <msclr\marshal_cppstd.h>

#include "Eval.h"

#using <System.dll>
#using <System.Collections.dll>

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Collections;

using namespace Microsoft::MSR::CNTK;

namespace Microsoft {    namespace MSR {        namespace CNTK 
{
    typedef void(*GetEvalProc)(IEvaluateModel<float>** peval);

    public ref class IEvaluateModelManaged
    {
    public:
        IEvaluateModelManaged(String^ config)
        {
            msclr::interop::marshal_context context;
            const std::string stdConfig = context.marshal_as<std::string>(config);

            pin_ptr<const WCHAR> dllname = PtrToStringChars("evaldll.dll");
            auto hModule = LoadLibrary(dllname);

            // create a variable of each type just to call the proper templated version
            auto procAddress = GetProcAddress(hModule, "GetEvalF");

            GetEvalProc getEvalProc = (GetEvalProc)procAddress;
            pin_ptr <IEvaluateModel<float>*> p_eval = &m_eval;
            getEvalProc(p_eval);

            m_eval->Init(stdConfig);

            pin_ptr<const WCHAR> stdModelPath = PtrToStringChars("E:\\VSO\\Source\\Repos\\CNTK_CUDA70\\Examples\\Other\\Simple2d\\Output\\Models\\simple.dnn");
            m_eval->LoadModel(stdModelPath);

            //m_eval->Evaluate()
        }

        void Init(String^ config)
        {

        }

        void Destroy()
        {
            
        }

        void LoadModel(String^ modelFileName)
        {

        }

        void Evaluate(Dictionary<String^, List<float>^> inputs, Dictionary<String^, List<float>^> outputs)
        {

            std::map<std::wstring, std::vector<float>*> stdInputs; // = new std::map<std::wstring, std::vector<float>*>();
            std::map<std::wstring, std::vector<float>*> stdOutputs;

            //TODO:

            m_eval->Evaluate(stdInputs, stdOutputs);
        }


    private:
        NodeGroup node;
        IEvaluateModel<float>** m_ImplFloat;
        IEvaluateModel<float> *m_eval;
    };
}}}