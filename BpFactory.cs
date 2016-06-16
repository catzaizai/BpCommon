using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BpCommon
{
    public class BpFactory
    {
        private static Bp _bp;

        public static void Initialization(int inputSize, int hiddenSize, int outputSize)
        {
            _bp = new Bp(inputSize, hiddenSize, outputSize);
        }
    }
}
