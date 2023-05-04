using BFLib.AI;
using System;
using System.Reflection;
using System.Text;

namespace Test
{
    internal class Program
    {
        public const int BATCH_SIZE = 3;

        static Random rand = new Random();

        static void Main(string[] args)
        {
            DenseNeuralNetwork network = new DenseNeuralNetwork(
                //new ActivationLayer(3, ActivationFunc.Tanh),
                //new BatchNormLayer(3, ForwardLayer.ForwardPort.In),
                3,
                new BatchNormLayer(3, ForwardLayer.ForwardPort.In),
                new BatchNormLayer(3, ForwardLayer.ForwardPort.Out),
                new ActivationLayer(3, ActivationFunc.Tanh),
                new BatchNormLayer(3, ForwardLayer.ForwardPort.Out),
                new ActivationLayer(3, ActivationFunc.Sigmoid),
                new BatchNormLayer(3, ForwardLayer.ForwardPort.Out),
                new ActivationLayer(3, ActivationFunc.Sigmoid),
                new BatchNormLayer(3, ForwardLayer.ForwardPort.Out),
                new ActivationLayer(3, ActivationFunc.Tanh),
                4
                );

            DenseNNForwardResult result;

            double[][] inputs =
            {
                new double[] { 1, 0.5, 0.6 },
                new double[] { 0.8, 0.4, 0.4 },
                new double[] { 0.8, 0.4, 0.6 },
                new double[] { 1, 0.4, 0.4 },
                new double[] { 0.8, 0.5, 0.6 }
            };

            double[][] desiredOutputs =
            {
                new double[] { 0.2, -0.9, 0.3, 0 },
                new double[] { 0.1, -0.5, 0.7, 0 },
                new double[] { 0.1, -0.5, 0.3, 0 },
                new double[] { 0.2, -0.5, 0.7, 0 },
                new double[] { 0.1, -0.9, 0.3, 0 }
            };

            network.BiasAssignForEach(RandomDouble);
            network.WeightAssignForEach(RandomDouble);

            for (int epoch = 0; epoch < 1000; epoch++)
            {
                int[] sampleIndexes = SampleIndex(0, inputs.Length, BATCH_SIZE);
                double[,] sampleOutputs = Sample(desiredOutputs, sampleIndexes);

                result = network.Forward(Sample(inputs, sampleIndexes));
                network.GradientDescent(sampleOutputs, result, 0.05 / BATCH_SIZE);

                if ((epoch + 1) % 10 == 0 || epoch == 0)
                {
                    Console.WriteLine(epoch + 1 + " run: ");
                    Console.Write(LogBatchNorm(network));
                    // Console.WriteLine("Network: ");
                    // Console.Write(ToString(network));
                    // LogOutput(result);

                    double error = 0;
                    for (int i = 0; i < sampleOutputs.GetLength(0); i++)
                        for (int j = 0; j < sampleOutputs.GetLength(1); j++)
                            error += Math.Pow(sampleOutputs[i, j] - result.outputs[i, j], 2);

                    Console.WriteLine();
                    Console.WriteLine("Error: " + error);
                    Console.WriteLine();
                    if (error < 0.001 / (sampleOutputs.Length * sampleOutputs.Length))
                        break;
                }

                //Console.WriteLine(epoch + 1 + " run: ");
                //Console.WriteLine("Network: ");
                //Console.Write(ToString(network));
                //LogOutput(result);
                //Console.WriteLine();

                //double error = 0;
                //for (int i = 0; i < sampleOutputs.GetLength(0); i++)
                //    for (int j = 0; j < sampleOutputs.GetLength(1); j++)
                //        error += Math.Pow(sampleOutputs[i, j] - result.outputs[i, j], 2);

                //if (error < 0.001 / (sampleOutputs.Length * sampleOutputs.Length))
                //    break;
                //Console.WriteLine("Error: " + error);
            }

            Console.ReadKey();
        }

        static int[] SampleIndex(int min, int max, int batchSize)
        {
            int[] result = new int[batchSize];
            int currentIndex = min;

            for (int randCount = 0; randCount < batchSize; randCount++) {
                currentIndex = rand.Next(currentIndex, max); // random false
                result[randCount] = currentIndex; 
            }

            return result;
        } 

        static double[,] Sample(double[][] population, params int[] indexes)
        {
            double[,] batch = new double[indexes.Length, population[0].Length];

            for(int index  = 0; index < indexes.Length; index++)
                for(int i = 0; i < population[index].Length; i++)
                    batch[index, i] = population[index][i];

            return batch;
        }

        static double[,] Sample(double[][] population, int batchSize)
        {
            double[,] batch = new double[batchSize, population[0].Length];

            int currentIndex = 0;
            for(int randCount = 0; randCount < batchSize; randCount++)
            {
                currentIndex += rand.Next(currentIndex, population.Length); // random false 

                for (int i = 0; i < population[currentIndex].Length; i++)
                    batch[currentIndex, i] = population[currentIndex][i];
            }

            return batch;
        }

        static void LogOutput(DenseNNForwardResult result)
        {
            double[,] outputs = result.outputs;

            for (int i = 0; i < outputs.GetLength(0); i++)
            {
                Console.Write(outputs[i, 0]);

                for (int j = 1; j < outputs.GetLength(1); j++)
                    Console.Write(" : " + outputs[i, j]);

                Console.WriteLine();
            }
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
                if(network.layers[i] is BatchNormLayer)
                {
                    sb.Append(String.Format("Gamma: {0}\t Beta: {1}", ((BatchNormLayer)network.layers[i]).gamma, ((BatchNormLayer)network.layers[i]).beta));
                    sb.AppendLine();
                }
            }

            return sb.ToString();
        }

        static string LogBatchNorm(DenseNeuralNetwork network)
        {
            StringBuilder sb = new StringBuilder();

            for (int i = 1; i < network.layers.Length; i++)
            {
                if(network.layers[i] is BatchNormLayer)
                {
                    sb.Append("Layer " + (i + 1) + ": ");
                    sb.Append(String.Format("Gamma: {0}\t Beta: {1}", ((BatchNormLayer)network.layers[i]).gamma, ((BatchNormLayer)network.layers[i]).beta));
                    sb.AppendLine();
                }
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