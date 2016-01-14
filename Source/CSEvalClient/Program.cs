using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Configuration;
using System.Text;
using System.Threading.Tasks;
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
            model.LoadModel("E:\\VSO\\Source\\Repos\\CNTK_CUDA70\\Examples\\Other\\Simple2d\\Output\\Models\\simple.dnn");
            var inputs = GetInputs();
            var outputs = GetOutputs();
            
            Console.WriteLine("Evaluating Model...");
            model.Evaluate(inputs, outputs);
            
            Console.WriteLine("Destroying Model...");
            model.Destroy();

            Console.WriteLine("Press <Enter> to terminate.");
            Console.ReadLine();
        }

        static Dictionary<string, List<float>> GetInputs()
        {
            string key1 = "key1";
            var inputs = new List<float>() {1, 0};

            return new Dictionary<string, List<float>>() {{ key1, inputs }};
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
