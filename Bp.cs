using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http.Headers;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace BpCommon
{
    [Serializable]
    public class Bp
    {
        private static readonly long SerialVersionUid = 1L;

        // input vector
        private readonly double[] _input;

        // hidden layer
        private readonly double[] _hidden;

        // output layer
        private readonly double[] _outPut;

        // target
        private readonly double[] _target;

        // delta vector of the hidden layer
        private readonly double[] _hidDelta;

        // output layer of the output layer
        private readonly double[] _optDelta;

        // learning rate
        private readonly double _eta;

        // momentum
        private readonly double _momentum;

        // weight matrix from input layer to hidden layer
        private readonly double[,] _iptHidWeights;

        // weight matrix from hidden layer to output layer
        private readonly double[,] _hidOptWeights;

        // previous weight update
        private readonly double[,] _iptHidPrevUptWeights;

        private readonly double[,] _hidOptPrevUptWeights;

        public double OptErrSum = 0d;

        public double HidErrSum = 0d;

        public readonly Random Random;

        public Bp(int inputSize, int hiddenSize, int outputSize, double eta, double momentum)
        {
            _input = new double[ inputSize + 1];
            _hidden = new double[hiddenSize + 1];
            _outPut = new double[outputSize + 1];
            _target = new double[outputSize + 1];

            _hidDelta = new double[hiddenSize + 1];
            _optDelta = new double[outputSize + 1];

            _iptHidWeights = new double[inputSize + 1, hiddenSize + 1];
            _hidOptWeights = new double[hiddenSize + 1, outputSize + 1];

            Random = new Random(DateTime.Now.Millisecond);
            RandomizeWeights(_iptHidWeights);
            RandomizeWeights(_hidOptWeights);

            _iptHidPrevUptWeights = new double[inputSize + 1, hiddenSize + 1];
            _hidOptPrevUptWeights = new double[hiddenSize + 1, outputSize + 1];

            _eta = eta;
            _momentum = momentum;
        }

        public Bp(int inputSize, int hiddenSize, int outputSize): this(inputSize, hiddenSize, outputSize, 0.998, 0.001)
        {
            
        }

        public void Train(double[] trainData, double[] target)
        {
            LoadInput(trainData);
            LoadTarget(target);
            Forward();
            CalculateDelta();
            AdjustWeight();
        }

        #region Private Method

        private void RandomizeWeights(double[,] matrix)
        {
            for (int i = 0, len = matrix.GetLength(0); i < len; i++)
            {
                for (int j = 0, len2 = matrix.GetLength(1); j < len2; j++)
                {
                    var real = Random.NextDouble();
                    matrix[i, j] = Random.NextDouble() > 0.5 ? real : -real;
                }
            }
        }

        private double[] GetNetworkOutput()
        {
            var len = _outPut.Length;
            var temp = new double[len - 1];
            for (int i = 1; i < len; i++)
            {
                temp[i - 1] = _outPut[i];
            }
            return temp;
        }

        /// <summary>
        /// Load the target data.
        /// </summary>
        /// <param name="arg"></param>
        private void LoadTarget(double[] arg)
        {
            if (arg.Length < _target.Length - 1)
            {
                throw new Exception("Size Do Not Match.");
            }
            Array.Copy(arg, 0, _target, 1, arg.Length);
        }

        /// <summary>
        /// Load the training data.
        /// </summary>
        /// <param name="inData"></param>
        private void LoadInput(double[] inData)
        {
            if (inData.Length != _input.Length)
            {
                throw new Exception("Size Do Not Match");
            }
            Array.Copy(inData, 0, _input, 1, inData.Length);
        }

        /// <summary>
        /// Forward
        /// </summary>
        /// <param name="layer0"></param>
        /// <param name="layer1"></param>
        /// <param name="weight"></param>
        private void Forward(double[] layer0, double[] layer1, double[,] weight)
        {
            layer0[0] = 1.0;
            for (var j = 0; j < layer1.Length; j++)
            {
                var sum = layer0.Select((t, i) => weight[i, j]*t).Sum();
                layer1[j] = Sigmoid(sum);
            }
            
        }

        /// <summary>
        /// Forward
        /// </summary>
        private void Forward()
        {
            Forward(_input, _hidden, _iptHidWeights);
            Forward(_hidden, _outPut, _hidOptWeights);
        }

        /// <summary>
        /// Calculate output error
        /// </summary>
        public void OutputErr()
        {
            var errSum = 0.0;
            for (int i = 1; i < _optDelta.Length; i++)
            {
                var o = _outPut[i];
                _optDelta[i] = o*(1d - o)*(_target[i] - o);
                errSum += Math.Abs(_optDelta[i]);
            }
            OptErrSum = errSum;
        }

        /// <summary>
        /// Calculate hidden errors
        /// </summary>
        private void HiddenErr()
        {
            var errSum = 0.0;
            for (var i = 1; i != _hidDelta.Length; i++)
            {
                var o = _hidDelta[i];
                var sum = _optDelta.Select((t, j) => _hidOptWeights[i, j]*t).Sum();
                _hidDelta[i] = o*(1d - o)*sum;
                errSum += Math.Abs(_hidDelta[i]);
            }
            HidErrSum = errSum;
        }

        /// <summary>
        /// Calculate errors of all layers
        /// </summary>
        private void CalculateDelta()
        {
            OutputErr();
            HiddenErr();
        }

        private void AdjustWeight(double[] delta, double[] layer, double[,] weight, double[,] prevWeight)
        {
            layer[0] = 1;
            for (var i = 1; i < delta.Length; i++)
            {
                for (var j = 0; j < layer.Length; j++)
                {
                    var newVal = _momentum*prevWeight[j, i] + _eta*delta[j]*layer[j];
                    weight[j, i] += newVal;
                    prevWeight[j, i] = newVal;
                }
            }
        }

        private void AdjustWeight()
        {
            AdjustWeight(_optDelta, _hidden, _hidOptWeights, _hidOptPrevUptWeights);
            AdjustWeight(_hidDelta, _input, _iptHidWeights, _iptHidPrevUptWeights);
        }

        private double Sigmoid(double val)
        {
            return 1d/(1d + Math.Exp(-val));
        }

        private double Tansig(double val)
        {
            return 2d/(1d + Math.Exp(-2*val)) - 1;
        }

        #endregion


    }
}
