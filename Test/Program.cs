using BFLib.AI;
using System.Text;

namespace Test
{
    internal class Program
    {
        static Random rand = new Random();

        static void Main(string[] args)
        {
            DenseNeuralNetwork network = new DenseNeuralNetwork(2, 3, 3, 2);
            DenseNNForwardResult result;

            double[] inputs =
            {
                2, 1
            };

            double[] desiredOutputs =
            {
                0.5, 1
            };

            network.BiasAssignForEach(RandomDouble);
            network.WeightAssignForEach(RandomDouble);

            for (int epoch = 0; epoch < 100; epoch++)
            {
                Console.WriteLine(epoch + 1 + " run: ");
                Console.WriteLine("Network: ");
                Console.Write(ToString(network));
                result = network.Forward(inputs);
                LogOutput(result);
                network.GradientDescent(desiredOutputs, result, 0.2);
                Console.WriteLine();
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
            for (int j = 0; j < network.layers[first].biases.Length; j++)
                sb.Append(network.layers[first].biases[j] + "\t");
            sb.AppendLine();

            for (int i = 1; i < network.layers.Length - 1; i++)
            {
                sb.AppendLine("Weight " + i + ": ");
                for (int j = 0; j < network.weights[i - 1].matrix.GetLength(0); j++)
                {
                    for (int k = 0; k < network.weights[i - 1].matrix.GetLength(1); k++)
                        sb.Append(network.weights[i - 1].matrix[j, k] + "\t");
                    sb.AppendLine();
                }

                sb.AppendLine("Layer " + (i + 1) + ": ");
                for (int j = 0; j < network.layers[i].biases.Length; j++)
                    sb.Append(network.layers[i].biases[j] + "\t");
                sb.AppendLine();
            }

            int last = network.layers.Length - 1;

            sb.AppendLine("Weight " + last + ": ");
            for (int j = 0; j < network.weights[last - 1].matrix.GetLength(0); j++)
            {
                for (int k = 0; k < network.weights[last - 1].matrix.GetLength(1); k++)
                    sb.Append(network.weights[last - 1].matrix[j, k] + "\t");
                sb.AppendLine();
            }

            return sb.ToString();
        }

        static double RandomDouble()
        {
            //return rand.Next(0, 1);
            return rand.NextDouble();
        }
    }
}