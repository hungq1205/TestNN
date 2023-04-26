﻿using BFLib.AI;
using System.Text;

namespace Test
{
    internal class Program
    {
        static Random rand = new Random();

        static void Main(string[] args)
        {
            DenseNeuralNetwork network = new DenseNeuralNetwork(
                3,
                new ActivationLayer(3, ActivationFunc.NaturalLog, false),
                new ActivationLayer(5, ActivationFunc.Tanh),
                new ActivationLayer(4, ActivationFunc.Tanh),
                new ActivationLayer(3, ActivationFunc.Tanh),
                //new ActivationLayer(5, ActivationFunc.Sigmoid),
                //new ActivationLayer(4, ActivationFunc.Sigmoid),
                //new ActivationLayer(3, ActivationFunc.Sigmoid),
                new ActivationLayer(3, ActivationFunc.Exponential, false),
                4
                );

            DenseNNForwardResult result;

            double[] inputs =
            {
                10000, 5, 500
            };

            double[] desiredOutputs =
            {
                20, -100, 100, 0
            };

            network.BiasAssignForEach(RandomDouble);
            network.WeightAssignForEach(RandomDouble);

            for (int epoch = 0; epoch < 1000; epoch++)
            {
                Console.WriteLine(epoch + 1 + " run: ");
                // Console.WriteLine("Network: ");
                // Console.Write(ToString(network));
                result = network.Forward(inputs);
                LogOutput(result);
                network.GradientDescent(desiredOutputs, result, 0.05);
                Console.WriteLine();

                double error = 0;
                for (int i = 0; i < desiredOutputs.LongLength; i++)
                    error += Math.Pow(desiredOutputs[i] - result.outputs[i], 2);

                if (error < 0.001 / (desiredOutputs.Length * desiredOutputs.Length))
                    break;
            }

            Console.ReadKey();
        }

        static void LogOutput(DenseNNForwardResult result)
        {
            double[] outputs = result.outputs;

            Console.Write("out: \n" + outputs[0]);
            for (int i = 1; i < outputs.Length; i++)
                Console.Write(" : " + outputs[i]);
            Console.WriteLine();
        }

        static string ToString(DenseNeuralNetwork network)
        {
            StringBuilder sb = new StringBuilder();

            int first = 0;

            sb.AppendLine("Layer " + (first + 1) + ": ");
            for (int j = 0; j < network.layers[first].dim; j++)
                sb.Append(network.layers[first].GetBias(j) + "\t");
            sb.AppendLine();

            for (int i = 1; i < network.layers.Length; i++)
            {
                sb.AppendLine("Weight " + i + ": ");
                for (int j = 0; j < network.weights[i - 1].outDim; j++)
                {
                    for (int k = 0; k < network.weights[i - 1].inDim; k++)
                        sb.Append(network.weights[i - 1].GetWeight(k, j) + "\t");
                    sb.AppendLine();
                }

                sb.AppendLine("Layer " + (i + 1) + ": ");
                for (int j = 0; j < network.layers[i].dim; j++)
                    sb.Append(network.layers[i].GetBias(j) + "\t");
                sb.AppendLine();
            }

            return sb.ToString();
        }

        static double RandomDouble()
        {
            //return rand.Next(1, 3);
            return (rand.Next(0, 2) * 2 - 1) * (rand.NextDouble() * 0.3d + 0.3d);
        }
    }
}