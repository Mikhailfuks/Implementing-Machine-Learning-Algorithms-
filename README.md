using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearningAlgorithms
{
    // Sample data class
    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength { get; set; }

        [LoadColumn(1)]
        public float SepalWidth { get; set; }

        [LoadColumn(2)]
        public float PetalLength { get; set; }

        [LoadColumn(3)]
        public float PetalWidth { get; set; }

        [LoadColumn(4), ColumnName("Label")]
        public string Species { get; set; }
    }

    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public string Species { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load data
            MLContext mlContext = new MLContext();
            string dataPath = "iris.csv"; // Replace with your data file path
            IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(dataPath, hasHeader: true, separatorChar: ',');

            // 2. Define training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            // 3. Train the model
            ITransformer model = pipeline.Fit(dataView);

            // 4. Create a prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<IrisData, Prediction>(model);

            // 5. Make a prediction
            IrisData newIris = new IrisData()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f
            };

            Prediction prediction = predictionEngine.Predict(newIris);

            // 6. Display the prediction
            Console.WriteLine($"Predicted Species: {prediction.Species}");

            Console.ReadKey();
        }
    }
}
