using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.MSR.CNTK;

namespace CSEvalClient
{
    class Program
    {
        static void Main(string[] args)
        {
            Environment.CurrentDirectory = Path.Combine(Environment.CurrentDirectory, @"..\..\Examples\Image\MNIST\Data\");
            Console.WriteLine("Current Directory: {0}", Environment.CurrentDirectory);

            Console.WriteLine("Creating Model Evaluator...");
            string config = GetConfig();

            var model = new IEvaluateModelManagedF();

            Console.WriteLine("Initializing Model Evaluator...");
            model.Init(config);

            Console.WriteLine("Loading Model...");

            string modelFilePath = Path.Combine(Environment.CurrentDirectory,
                @"..\Output\Models\01_OneHidden");

            model.LoadModel(modelFilePath);

            var inputs = GetDictionary("features", 28 * 28, 255);
            var outputs = GetDictionary("ol.z", 10, 100);

            Console.WriteLine("Evaluating Model...");
            model.Evaluate(inputs, outputs);

            Console.WriteLine("Destroying Model...");
            model.Destroy();

            foreach (var item in outputs.First().Value)
            {
                Console.WriteLine(item);
            }

            Console.WriteLine("Press <Enter> to terminate.");
            Console.ReadLine();
        }

        static Dictionary<string, List<float>> GetDictionary(string key, int size, int maxValue)
        {
            return new Dictionary<string, List<float>>() { { key, GetFloatArray(size, maxValue) } };
        }

        static string GetConfig()
        {
            string configFilePath = Path.Combine(Environment.CurrentDirectory,
                    @"..\Config\01_OneHidden.config");

            var lines = System.IO.File.ReadAllLines(configFilePath);
            return string.Join("\n", lines);
        }

        static List<float> GetFloatArray(int size, int maxValue)
        {
            Random rnd = new Random();
            return Enumerable.Range(1, size).Select(i => (float)rnd.Next(maxValue)).ToList();
        }
    }
}
