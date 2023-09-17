using BFLib.AI;
using BFLib.Data;
using BFLib.Utility.Linear;
using System;
using System.Reflection;
using System.Text;

namespace Test
{
    internal class Program
    {
        public const int BATCH_SIZE = 1;

        static Random rand = new Random();

        static int dataLength;

        static void Main(string[] args)
        {
            //NeuralNetworkTest();
            LinearTest();

            Console.ReadKey();
        }

        static void LinearTest()
        {
            while (true)
            {
                // SquareMatrix A = SquareMatrix.Diag(2, 3, 6, 4);
                SquareMatrix A = RandomMatrix(4, 1, 6);
                A = A * A.Transpose;
                Vector b = RandomVector(4, 1, 6);
                Vector X = A.Invert() * b;
                Vector cg;

                Console.WriteLine("A: ");
                PrintMatrix(A);
                Console.WriteLine("A * At: ");
                PrintMatrix(A * A.Transpose);
                Console.WriteLine("b: ");
                PrintVector(b);
                Console.WriteLine("X: ");
                PrintVector(X);

                cg = LinearMethod.CGMethod(A, b);
                Console.WriteLine("X (CG): ");
                PrintVector(cg);

                Console.ReadKey(true);
                Console.WriteLine();
                Console.WriteLine();
            }
        }

        static void NeuralNetworkTest()
        {
            double[][] inputs =
            {
                new double[] { 0, 1.00d, 0.4d, 1 },
                new double[] { 1, 0.75d, 0.6d, 0 },
                new double[] { 0, 0.50d, 0.8d, 0 },
                new double[] { 1, 0.50d, 0.6d, 1 },
                new double[] { 0, 0.25d, 0.8d, 1 },
                new double[] { 1, 1.00d, 0.4d, 1 },
                new double[] { 0, 0.50d, 0.6d, 1 },
                new double[] { 1, 0.25d, 0.8d, 0 },
                new double[] { 0, 0.50d, 0.4d, 1 },
                new double[] { 1, 0.00d, 0.4d, 1 },
            };

            double[][] desiredOutputs =
            {
                new double[] { 1.00d / 2, 1.4d / 2},
                new double[] { 1.75d / 2, 0.6d / 2},
                new double[] { 0.50d / 2, 0.8d / 2},
                new double[] { 1.50d / 2, 1.6d / 2},
                new double[] { 0.25d / 2, 1.8d / 2},
                new double[] { 2.00d / 2, 1.4d / 2},
                new double[] { 0.50d / 2, 1.6d / 2},
                new double[] { 1.25d / 2, 0.8d / 2},
                new double[] { 0.50d / 2, 1.4d / 2},
                new double[] { 1.00d / 2, 1.4d / 2},
            };

            DenseNeuralNetworkBuilder builder = new DenseNeuralNetworkBuilder(inputs[0].Length);
            builder.NewLayers(
                //new ActivationLayer(3, ActivationFunc.Tanh),
                //new BatchNormLayer(3, ForwardLayer.ForwardPort.In),
                //new BatchNormLayer(ForwardLayer.ForwardPort.In),
                new ActivationLayer(12, ActivationFunc.Linear),
                new ActivationLayer(12, ActivationFunc.Linear),
                //new BatchNormLayer(ForwardLayer.ForwardPort.In),
                //new ActivationLayer(6, ActivationFunc.Linear),
                //new BatchNormLayer(ForwardLayer.ForwardPort.In),
                new ActivationLayer(desiredOutputs[0].Length, ActivationFunc.Sigmoid)
                );

            //Optimizer optimizer = new SGD(0.03d);
            //Optimizer optimizer = new AdaGrad(0.1d);
            //Optimizer optimizer = new AdaDelta(0.9d);
            //Optimizer optimizer = new Momentum(0.9d, 0.01d);
            //Optimizer optimizer = new RMSprop(0.99d);
            Optimizer optimizer = new Adam(0.9d, 0.99d, 0.01d);
            DenseNeuralNetwork network = new DenseNeuralNetwork(builder, optimizer);

            //RecurrentNeuralNetwork network = new RecurrentNeuralNetwork(3, 1, 1);

            //string[] cats = UData.GetCategoriesFromCSV(@"C:\Users\mAy tInH 2 mAn HiNh\Documents\datasets\realtor-data.csv");
            //Dictionary<string, AdditionNumericDataInfo> inputInfos, outputInfos;
            //DistinctIntDataInfo[] distinctInfos;
            //UDataInfo info = new UDataInfo(
            //    new string[]
            //    {
            //        "status",
            //        "zip_code",
            //        "prev_sold_date",
            //        "price"
            //    },
            //    DataType.Neglect,
            //    DataType.Double,
            //    DataType.Double,
            //    DataType.Double,
            //    DataType.DistinctInt,
            //    DataType.DistinctInt,
            //    DataType.Neglect,
            //    DataType.Double,
            //    DataType.Neglect,
            //    DataType.Neglect
            //    );
            //Dictionary<string, double>[] inputs = UData.RetrieveUDataFromCSV(@"C:\Users\mAy tInH 2 mAn HiNh\Documents\datasets\realtor-data.csv", info, out distinctInfos, out inputInfos, 100);
            //Dictionary<string, double>[] inputs = UData.RetrieveNumberDataFromCSV(@"C:\Users\mAy tInH 2 mAn HiNh\Documents\datasets\realtor-data.csv", 5000, out inputInfos, "bed", "house_size");
            //Dictionary<string, double>[] desiredOutputs = UData.RetrieveNumberDataFromCSV(@"C:\Users\mAy tInH 2 mAn HiNh\Documents\datasets\realtor-data.csv", 5000, out outputInfos, "price");

            ForwardResult result;

            dataLength = inputs.Length;

            network.BiasAssignForEach(RandomDouble);
            network.WeightAssignForEach(RandomDouble);

            for (int epoch = 0; epoch < 1000000; epoch++)
            {
                int[] sampleIndexes = SampleIndex(0, dataLength, BATCH_SIZE);
                //double[][] sampleOutputs = ToDoubleArrays(Sample(desiredOutputs, sampleIndexes), outputInfos);

                double[][] sampleOutputs = new double[sampleIndexes.Length][];
                for (int i = 0; i < sampleIndexes.Length; i++)
                    sampleOutputs[i] = desiredOutputs[sampleIndexes[i]];

                result = network.Forward(Sample(inputs, sampleIndexes));

                network.GradientDescent(sampleOutputs, result);

                if ((epoch + 1) % 100000 == 0 || epoch == 0)
                {
                    Console.WriteLine(epoch + 1 + " run: ");
                    // Console.Write(LogBatchNorm(network));
                    // Console.Write(ToString(network));
                    Console.WriteLine(NonForwardToString(network));
                    // LogOutput(result);
                    LogCompareOutput(result, sampleOutputs);

                    double error = 0;
                    for (int i = 0; i < sampleOutputs.Length; i++)
                        for (int j = 0; j < sampleOutputs[0].Length; j++)
                            error += Math.Pow(sampleOutputs[i][j] - result.outputs[i][j], 2);

                    Console.WriteLine();
                    Console.WriteLine("Error: " + error);
                    Console.WriteLine();
                    //if (error < 0.001 / (sampleOutputs.Length * sampleOutputs.Length))
                    //    break;
                }
            }

            Console.ReadKey();
            while (true)
            {
                for (int epoch = 0; epoch < 1; epoch++)
                {
                    int[] sampleIndexes = SampleIndex(0, dataLength, 1);
                    //int[] sampleIndexes = SampleIndex(0, dataLength, BATCH_SIZE);
                    //double[][] sampleOutputs = ToDoubleArrays(Sample(desiredOutputs, sampleIndexes), outputInfos);

                    double[][] sampleOutputs = new double[sampleIndexes.Length][];
                    for (int i = 0; i < sampleIndexes.Length; i++)
                        sampleOutputs[i] = desiredOutputs[sampleIndexes[i]];

                    result = network.Forward(Sample(inputs, sampleIndexes));
                    network.GradientDescent(sampleOutputs, result);

                    if ((epoch + 1) % 1 == 0 || epoch == 0)
                    {
                        Console.WriteLine(epoch + 1 + " run: ");
                        //Console.WriteLine(LogBatchNorm(network));
                        // Console.WriteLine(ToString(network));
                        //Console.WriteLine(NonForwardToString(network));
                        //Console.WriteLine(LogForwardLog(result, network.layers));
                        // LogOutput(result);
                        LogCompareOutput(result, sampleOutputs);

                        double error = 0;
                        for (int i = 0; i < sampleOutputs.Length; i++)
                            for (int j = 0; j < sampleOutputs[0].Length; j++)
                                error += Math.Pow(sampleOutputs[i][j] - result.outputs[i][j], 2);

                        Console.WriteLine();
                        Console.WriteLine("Error: " + error);
                        Console.WriteLine();
                        //if (error < 0.001 / (sampleOutputs.Length * sampleOutputs.Length))
                        //    break;
                    }
                }
                Console.ReadKey();
            }
        }

        static SquareMatrix RandomMatrix(int dim, double min = 0, double max = 1)
        {
            double[][] content = new double[dim][];

            for (int i = 0; i < dim; i++) {
                content[i] = new double[dim];

                for (int j = 0; j < dim; j++)
                    content[i][j] = rand.NextDouble() * (max - min) + min;
            }

            return content;
        }

        static SquareMatrix RandomMatrix(int dim, int min = 0, int max = 2)
        {
            double[][] content = new double[dim][];

            for (int i = 0; i < dim; i++) {
                content[i] = new double[dim];

                for (int j = 0; j < dim; j++)
                    content[i][j] = rand.Next(min, max);
            }

            return content;
        }

        static Vector RandomVector(int dim, double min = 0, double max = 1)
        {
            double[] content = new double[dim];

            for (int i = 0; i < dim; i++)
                    content[i] = rand.NextDouble() * (max - min) + min;

            return content;
        }

        static Vector RandomVector(int dim, int min = 0, int max = 2)
        {
            double[] content = new double[dim];

            for (int i = 0; i < dim; i++)
                    content[i] = rand.Next(min, max);

            return content;
        }

        static void PrintMatrix(SquareMatrix matrix)
        {
            for (int i = 0; i < matrix.dim; i++)
            {
                Console.Write("|");
                Console.Write("{0,5: #0.0;-#0.0; #0.0}", matrix.content[i][0]);
                for (int j = 1; j < matrix.dim; j++)
                    Console.Write(" {0,5: #0.0;-#0.0; #0.0}", matrix.content[i][j]);
                Console.Write(" |\n");
            }
        }

        static void PrintVector(Vector vector)
        {
            for (int i = 0; i < vector.dim; i++)
                Console.WriteLine("|{0,5: #0.0;-#0.0; #0.0} |", vector.content[i]);
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

        //static IForwardInput[] Sample(double[][][] population, params int[] indexes)
        //{
        //    IForwardInput[] batch = new IForwardInput[indexes.Length];

        //    for(int i = 0; i < indexes.Length; i++)
        //        batch[i] = new IForwardInput(population[indexes[i]]);

        //    return batch;
        //}

        //static IForwardInput[] Sample(double[][][] population, int batchSize)
        //{
        //    return Sample(population, SampleIndex(0, population.Length, batchSize));
        //}

        static IForwardInput[] Sample(double[][] population, params int[] indexes)
        {
            IForwardInput[] batch = new IForwardInput[indexes.Length];

            for(int i = 0; i < indexes.Length; i++)
                batch[i] = new DenseForwardInput(population[indexes[i]]);

            return batch;
        }

        static IForwardInput[] Sample(double[][] population, int batchSize)
        {
            return Sample(population, SampleIndex(0, population.Length, batchSize));
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

        static double[][] ToDoubleArrays(Dictionary<string, double>[] content, Dictionary<string, AdditionNumericDataInfo> infos)
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
                    if (infos.ContainsKey(labels[i]))
                        result[sample][i] = infos[labels[i]].Normalize(NormalizationMode.DivideMean, content[sample][labels[i]]);
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

        static void LogOutput(ForwardResult result)
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

        static void LogCompareOutput(ForwardResult result, double[][] desired)
        {
            for (int i = 0; i < result.outputs.Length; i++)
                for (int j = 0; j < result.outputs[i].Length; j++)
                    Console.WriteLine(desired[i][j] + "\t:\t" + result.outputs[i][j]);
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

        static string LogForwardLog(DenseForwardResult result, Layer[] layers)
        {
            StringBuilder sb = new StringBuilder();

            for (int sample = 0; sample < result.outputs.Length; sample++)
            {
                for (int i = 0; i < result.layerInputs.Length; i++)
                {
                    if (layers[i] is ForwardLayer)
                        continue;
                    sb.AppendLine("Layer " + (i + 1) + ": ");
                    for (int j = 0; j < result.layerInputs[i][sample].Length; j++)
                        sb.Append(result.layerInputs[i][sample][j] + "\t");
                    sb.AppendLine();
                }

                sb.AppendLine();
            }

            return sb.ToString();
        }

        static double RandomDouble(double value)
        {
            //return rand.Next(1, 3);
            return (rand.Next(0, 2) * 2 - 1) * (rand.NextDouble() * 0.3d + 0.1d);
            //return (rand.NextDouble() * 0.3d + 0.1d);
        }
    }
}