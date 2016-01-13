using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.MSR.CNTK;

namespace CSEvalClient
{
    class Program
    {
        static void Main(string[] args)
        {
            string config = GetConfig();
            IEvaluateModelManaged model = new IEvaluateModelManaged(config);
            model.Destroy();
        }

        static string GetConfig()
        {
            var lines = System.IO.File.ReadAllLines(@"E:\VSO\Source\Repos\CNTK_CUDA70\Examples\Other\Simple2d\Config\Simple.config");
            return string.Join("", lines);
        }
    }
}
