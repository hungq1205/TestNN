using BFLib.AI;
using BFLib.Data;
using System;
using System.Reflection;
using System.Text;

namespace Test
{
    internal class Program
    {
        public const int BATCH_SIZE = 8;

        static Random rand = new Random();

        static int dataLength;

        static void Main(string[] args)
        {
            DenseNeuralNetwork network = new DenseNeuralNetwork(
                //new ActivationLayer(3, ActivationFunc.Tanh),
                //new BatchNormLayer(3, ForwardLayer.ForwardPort.In),
                6,
                new BatchNormLayer(6, ForwardLayer.ForwardPort.In), 
                new BatchNormLayer(4, ForwardLayer.ForwardPort.Out),
                new ActivationLayer(4, ActivationFunc.Tanh),
                new BatchNormLayer(4, ForwardLayer.ForwardPort.Out),
                new ActivationLayer(4, ActivationFunc.Tanh),
                new BatchNormLayer(3, ForwardLayer.ForwardPort.Out),
                new ActivationLayer(3, ActivationFunc.Tanh),
                1
                );

            string[] cats = UData.GetCategoriesFromCSV(@"C:\Users\mAy tInH 2 mAn HiNh\Documents\datasets\realtor-data.csv");
            Dictionary<string, Tuple<double, double>> inputRanges, outputRanges;
            DistinctIntDataInfo[] distinctInfos;
            UDataInfo info = new UDataInfo(
                new string[]
                {
                    "status",
                    "zip_code",
                    "prev_sold_date",
                    "price"
                },
                DataType.Neglect,
                DataType.Double,
                DataType.Double,
                DataType.Double,
                DataType.DistinctInt,
                DataType.DistinctInt,
                DataType.Neglect,
                DataType.Double,
                DataType.Neglect,
                DataType.Neglect
                );
            Dictionary<string, double>[] inputs = UData.RetrieveUDataFromCSV(@"C:\Users\mAy tInH 2 mAn HiNh\Documents\datasets\realtor-data.csv", info, out distinctInfos, out inputRanges, 50000);
            Dictionary<string, double>[] desiredOutputs = UData.RetrieveNumberDataFromCSV(@"C:\Users\mAy tInH 2 mAn HiNh\Documents\datasets\realtor-data.csv", 50000, out outputRanges, "price");

            dataLength = inputs.Length;

            DenseNNForwardResult result;

            //double[][] inputs =
            //{
            //    new double[] { 1, 0.5, 0.6 },
            //    new double[] { 0.8, 0.4, 0.4 },
            //    new double[] { 0.8, 0.4, 0.6 },
            //    new double[] { 1, 0.4, 0.4 },
            //    new double[] { 0.8, 0.5, 0.6 }
            //};

            //double[][] desiredOutputs =
            //{
            //    new double[] { 0.2, -0.9, 0.3, 0 },
            //    new double[] { 0.1, -0.5, 0.7, 0 },
            //    new double[] { 0.1, -0.5, 0.3, 0 },
            //    new double[] { 0.2, -0.5, 0.7, 0 },
            //    new double[] { 0.1, -0.9, 0.3, 0 }
            //};

            network.BiasAssignForEach(RandomDouble);
            network.WeightAssignForEach(RandomDouble);

            for (int epoch = 0; epoch < 30000; epoch++)
            {
                int[] sampleIndexes = SampleIndex(0, dataLength, BATCH_SIZE);
                double[][] sampleOutputs = ToDoubleArrays(Sample(desiredOutputs, sampleIndexes), outputRanges);

                result = network.Forward(Sample(inputs, sampleIndexes));
                network.GradientDescent(sampleOutputs, result, 0.05 / BATCH_SIZE);

                if ((epoch + 1) % 2000 == 0 || epoch == 0)
                {
                    Console.WriteLine(epoch + 1 + " run: ");
                    // Console.Write(LogBatchNorm(network));
                    // Console.WriteLine("Network: ");
                    // Console.Write(ToString(network));
                    // Console.Write(NonForwardToString(network));
                    // LogOutput(result);
                    LogCompareOutput(result, ToDoubleArrays(Sample(desiredOutputs, sampleIndexes)), outputRanges["price"]);

                    double error = 0;
                    for (int i = 0; i < sampleOutputs.Length; i++)
                        for (int j = 0; j < sampleOutputs[0].Length; j++)
                            error += Math.Pow(sampleOutputs[i][j] - result.outputs[i][j], 2);

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

        static int[] SampleIndex(int min, int max, int batchSize, int reservedPoint = int.MinValue)
        {
            int[] result = new int[batchSize];

            for (int i = 0; i < batchSize; i++)
                result[i] = reservedPoint;

            Random rand = new Random();

            for (int randCount = 0; randCount < batchSize; randCount++)
            {
                result[randCount] = rand.Next(min, max - randCount);

                int i = 0;
                do
                {
                    i = 0;
                    for (; i < randCount; i++)
                    {
                        if (result[randCount] == result[i])
                        {
                            result[randCount] = max - 1 - i;
                            i = -1;
                            break;
                        }
                    }
                } while (i == -1);
            }

            return result.ToArray();
        }

        static double[][] Sample(double[][] population, params int[] indexes)
        {
            double[][] batch = new double[indexes.Length][];

            for(int i = 0; i < indexes.Length; i++)
                batch[i] = population[indexes[i]];

            return batch;
        }

        static double[][] Sample(double[][] population, int batchSize)
        {
            double[][] batch = new double[batchSize][];

            int currentIndex = 0;
            for(int randCount = 0; randCount < batchSize; randCount++)
            {
                currentIndex += rand.Next(currentIndex, population.Length); // random false 

                batch[randCount] = population[currentIndex];
            }

            return batch;
        }

        static double[][] ToDoubleArrays(Dictionary<string, double>[] content)
        {
            List<string> keyList = new List<string>();
            foreach(string label in content[0].Keys)
                keyList.Add(label);

            double[][] result = new double[content.Length][];
            string[] keys = keyList.ToArray();

            for (int sample = 0; sample < content.Length; sample++)
            {
                result[sample] = new double[keys.Length];
                for (int i = 0; i < keys.Length; i++)
                    result[sample][i] = content[sample][keys[i]];
            }

            return result;
        }

        static double[][] ToDoubleArrays(Dictionary<string, double>[] content, Dictionary<string, Tuple<double, double>> ranges)
        {
            List<string> keyList = new List<string>();
            foreach(string label in content[0].Keys)
                keyList.Add(label);

            double[][] result = new double[content.Length][];
            string[] labels = keyList.ToArray();

            for (int sample = 0; sample < content.Length; sample++)
            {
                result[sample] = new double[labels.Length];
                for (int i = 0; i < labels.Length; i++)
                {
                    Tuple<double, double> range = ranges[labels[i]];

                    if (range != null)
                        result[sample][i] = (content[sample][labels[i]] - range.Item1) / (range.Item2 - range.Item1);
                    else
                        result[sample][i] = content[sample][labels[i]];
                }
            }

            return result;
        }

        static Dictionary<string, double>[] Sample(Dictionary<string, double>[] population, params int[] indices)
        {
            Dictionary<string, double>[] batch = new Dictionary<string, double>[indices.Length];

            for (int i = 0; i < indices.Length; i++)
            {
                batch[i] = new Dictionary<string, double>();

                foreach (string label in population[0].Keys)
                    batch[i].Add(label, population[indices[i]][label]);
            }

            return batch;
        }

        static Dictionary<string, double>[] Sample(Dictionary<string, double>[] population, int batchSize)
        {
            return Sample(population, SampleIndex(0, dataLength, batchSize));
        }

        static void LogOutput(DenseNNForwardResult result)
        {
            double[][] outputs = result.outputs;

            for (int i = 0; i < outputs.Length; i++)
            {
                Console.Write(outputs[i][0]);

                for (int j = 1; j < outputs[i].Length; j++)
                    Console.Write(" : " + outputs[i][j]);

                Console.WriteLine();
            }
        }

        static void LogCompareOutput(DenseNNForwardResult result, double[][] desired, Tuple<double, double> range)
        {
            double[][] outputs = result.outputs;

            for (int i = 0; i < outputs.Length; i++)
                for (int j = 0; j < outputs[i].Length; j++)
                    Console.WriteLine(NormRange(desired[i][j],range) + " : " + outputs[i][j], range);
                    //Console.WriteLine(desired[i][j],range + " : " + DenormRange(outputs[i][j], range));
        }

        static double DenormRange(double normed, Tuple<double, double> range) => normed * (range.Item2 - range.Item1) + range.Item1;

        static double NormRange(double normed, Tuple<double, double> range) => (normed - range.Item1) / (range.Item2 - range.Item1);

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

        static string NonForwardToString(DenseNeuralNetwork network)
        {
            StringBuilder sb = new StringBuilder();

            int first = 0;

            sb.AppendLine("Layer " + (first + 1) + ": ");
            for (int j = 0; j < network.layers[first].dim; j++)
                sb.Append(network.layers[first].GetBias(j) + "\t");
            sb.AppendLine();

            for (int i = 1; i < network.layers.Length; i++)
            {
                if (!(network.weights[i - 1] is ForwardWeightMatrix))
                {
                    sb.AppendLine("Weight " + i + ": ");
                    for (int j = 0; j < network.weights[i - 1].outDim; j++)
                    {
                        for (int k = 0; k < network.weights[i - 1].inDim; k++)
                            sb.Append(network.weights[i - 1].GetWeight(k, j) + "\t");
                        sb.AppendLine();
                    }
                }

                if (!(network.layers[i] is ForwardLayer))
                {
                    sb.AppendLine("Layer " + (i + 1) + ": ");
                    for (int j = 0; j < network.layers[i].dim; j++)
                        sb.Append(network.layers[i].GetBias(j) + "\t");
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
            return (rand.Next(0, 2) * 2 - 1) * (rand.NextDouble() * 0.3d + 0.1d);
        }
    }
}