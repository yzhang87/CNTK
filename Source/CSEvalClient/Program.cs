using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.MSR.CNTK;

namespace CSEvalClient
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Creating Model Evaluator...");
            string config = GetConfig();

            var model = new IEvaluateModelManagedF();

            Console.WriteLine("Initializing Model Evaluator...");
            model.Init(config);

            Console.WriteLine("Loading Model...");

            string modelFilePath = Path.Combine(Environment.CurrentDirectory,
                @"..\..\Examples\Other\Simple2d\Output\Models\simple.dnn");
            Console.WriteLine("Current Directory: '{0}'", Environment.CurrentDirectory);
            model.LoadModel(modelFilePath);
            var inputs = GetInputs();
            Dictionary<string, List<float>> outputs = new Dictionary<string, List<float>>() { { "", new List<float>() { 0, 0 } } };

            Console.WriteLine("Evaluating Model...");
            model.Evaluate(inputs, outputs);

            Console.WriteLine("Destroying Model...");
            model.Destroy();

            Console.WriteLine("Press <Enter> to terminate.");
            Console.ReadLine();
        }

        static Dictionary<string, List<float>> GetInputs()
        {
            string key1 = "features";
            var inputs = new List<float>() { 1, 0 };

            return new Dictionary<string, List<float>>() { { key1, inputs } };
        }

        static Dictionary<string, List<float>> GetOutputs()
        {
            string key1 = "key1";
            var outputs = new List<float>() { 1, 0 };

            return new Dictionary<string, List<float>>() { { key1, outputs } };
        }

        static string GetConfig()
        {
            var lines = System.IO.File.ReadAllLines(@"E:\VSO\Source\Repos\CNTK_CUDA70\Examples\Other\Simple2d\Config\Simple.config");
            return string.Join("", lines);
        }
    }
}
