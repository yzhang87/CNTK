// CMTestClient.cpp : Hello world for CNTK API
//

#include "stdafx.h"

#include <vector>

#include <ComputationNetwork.h>
#include <ComputationNetworkBuilder.h>
#include <EvalReader.h>
#include <Matrix.h>
#include <MPIWrapper.h>
#include <SGD.h>

using namespace Microsoft::MSR::CNTK;

// Set globals
bool g_shareNodeValueMatrices = false;
Microsoft::MSR::CNTK::MPIWrapper *g_mpi = nullptr;

typedef std::vector<float> FV;
typedef std::map<std::wstring, FV*> FVMapT;

int _tmain(int /* argc */, _TCHAR* /* argv[] */)
{
    // Define some training and test set
    std::vector<float> trainingInput, trainingLabels, testInput, testLabels;
    srand(0);

    for (int i = 0; i < 4000; ++i)
    {
        float x = (float)rand() / RAND_MAX, y = (float)rand() / RAND_MAX;
        float l = x > y / 2.f + 0.2f ? 1.f : -1.f;  // Function to learn
        if (i % 10 == 0)
        {
            testInput.push_back(x);  testInput.push_back(y); testLabels.push_back(l);
        }
        else
        {
            trainingInput.push_back(x);  trainingInput.push_back(y); trainingLabels.push_back(l);
        }

    }

    // Format it the way the reader needs it. The size map needs to contain the number of _rows_.
    FVMapT trainingSet = { { L"Inputs", &trainingInput }, { L"Labels", &trainingLabels } };
    std::map<std::wstring, size_t> trainingSizes = { { L"Inputs", 2 }, { L"Labels", 1 } };
    FVMapT testSet = { { L"Inputs", &testInput }, { L"Labels", &testLabels } };
    std::map<std::wstring, size_t> testSizes = { { L"Inputs", 2 }, { L"Labels", 1 } };


    try
    {
        // nn = sigmoid(W * I + b)
        ComputationNetworkPtr cn(new ComputationNetwork(CPUDEVICE));
        ComputationNetworkBuilder<float> builder(*cn);
        auto I = builder.CreateInputNode(L"Inputs", TensorShape(1, 2));
        auto L = builder.CreateInputNode(L"Labels", TensorShape(1));
        auto W = builder.CreateLearnableParameter(L"W", TensorShape(1, 2));
        auto b = builder.CreateLearnableParameter(L"b", TensorShape(1, 1));

        auto S = builder.Sigmoid( builder.Plus(builder.Times(W, I, L"Times"), b, L"Plus"), L"Sigmoid");
        auto C = builder.CrossEntropyWithSoftmax(L, S, L"CrossEnt");
        cn->FeatureNodes().push_back(I);
        cn->LabelNodes().push_back(L);
        cn->OutputNodes().push_back(S);
        cn->FinalCriterionNodes().push_back(C);
        cn->CompileNetwork();

        // Random initialization
        cn->InitLearnableParameters<float>(W, true, 1, 1.0);
        cn->InitLearnableParameters<float>(b, true, 2, 1.0);

        // Initialize SGD. Unfortunately this requires from us to go through the config parser
        // as there's no other constructor.
        ConfigParameters config;

        std::vector<wchar_t*> av = {
            L"[",
            L"modelPath=\"model.dat\"",
            L"SGD = [",
            L"  minibatchSize = 1000",
            L"  prefetchTrainingData = true",
            L"  epochSize = 0",
            L"  learningRatesPerMB = 0.8",
            L"  momentumPerMB = 0.9",
            L"  maxEpochs = 5",
            L"  minibatchSize = 256",
            L"]",
            L"# Reader doesn't use config, but requires config object as well.",
            L"Reader []",
            L"]" };
        std::string rawConfigString = ConfigParameters::ParseCommandLine(av.size(), &*av.begin(), config); // could also use config.FileParse(wstring)

        SGD<float> optimizer(ConfigParameters(config("SGD")));
        EvalReader<float> trainSetReader(config("Reader"));
        trainSetReader.SetData(&trainingSet, &trainingSizes);
        trainSetReader.SetBoundary(0);
        trainSetReader.SetSingleFrameMode(true); // No sequences

        EvalReader<float> validationSetReader(config("Reader"));
        validationSetReader.SetData(&testSet, &testSizes);
        validationSetReader.SetBoundary(0);
        validationSetReader.SetSingleFrameMode(true); // No sequences

        // Train takes a factory method which is only used if it figure it can't/shouldn't continue train
        optimizer.Train([cn](int /*deviceId*/) { return cn; }, CPUDEVICE, &trainSetReader, &validationSetReader, false);

        printf("FINAL PARAMETERS");
        W->Value().Print("Weights");
        b->Value().Print("Bias");
        cn->DumpAllNodesToFile(true, L"dump.post.txt");
        cn->PlotNetworkTopology(L"network.post.dot"); // Needs compiled network
        cn->Save(L"model.post.txt", FileOptions::fileOptionsText);

        // Do some evaluation ourselves

        validationSetReader.StartMinibatchLoop(testInput.size(), 0, 0);
        size_t mbSize;

        std::map<std::wstring, Matrix<float>*> inputs{ { L"Inputs", &I->Value() }, { L"Labels", &L->Value() } };
        DataReaderHelpers::GetMinibatchIntoNetwork(validationSetReader, cn, C, false, false, inputs, mbSize);
        validationSetReader.CopyMBLayoutTo(cn->GetMBLayoutPtr());
        cn->ForwardProp(cn->OutputNodes()[0]);

        for (int i = 0; i < testLabels.size(); ++i)
        {
            printf("%d: Expected: %f   Actual: %f\n", i, testLabels[i], S->Value()(0, i));
        }
    }
    catch (const std::exception& e)
    {
        printf("Exception: %s\n", e.what());
    }

    printf("Enter something>\n");
    int d;
    scanf_s("%d", &d);
    return 0;
}
