//Логистическая регрессия. Её хорошо использовать для задач бинарной классификации
//(это задачи, в которых на выходе мы получаем один из двух классов).

open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data

[<CLIMutable>]
type SpamInput = {
    [<LoadColumn(0)>] Spam : string
    [<LoadColumn(1)>] Text : string
}

[<CLIMutable>]
type SpamPrediction = {
    [<ColumnName("PredictedLabel")>] IsSpam : bool
    Score : float32
    Probability : float32
}

/// Output
[<CLIMutable>]
type ToLabel ={
    mutable Label : bool
}

/// Helper function
let castToEstimator (x : IEstimator<_>) = 
    match x with 
    | :? IEstimator<ITransformer> as y -> y
    | _ -> failwith "Cannot cast pipeline to IEstimator<ITransformer>"

/// dataset
let dataPath = sprintf "%s\\data\\spam.tsv" Environment.CurrentDirectory

[<EntryPoint>]
let main arv =
    
    //Context
    let context = new MLContext()
    //Load data
    let data = context.Data.LoadFromTextFile<SpamInput>(dataPath, hasHeader = true, separatorChar = '\t')

    // 15% for testing
    let partitions = context.Data.TrainTestSplit(data, testFraction = 0.15)

    let pipeline = 
        EstimatorChain()
            // transform the 'spam' and 'ham' values to true and false
            .Append(
                context.Transforms.CustomMapping(
                    Action<SpamInput, ToLabel>(fun input output -> output.Label <- input.Spam = "spam"),
                    "MyLambda"))

            // featureize the input text
            .Append(context.Transforms.Text.FeaturizeText("Features", "Text"))

            // use a stochastic dual coordinate ascent learner
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression())

    // training
    let model = partitions.TrainSet |> pipeline.Fit

    // set up a prediction engine
    let engine = context.Model.CreatePredictionEngine model

    // sample
    let messages = [
        { Text = "Hi, are you free today??"; Spam = "" }
        { Text = "WIN a IPHONE EVERY DAY. Txt WINNER now to join"; Spam = "" }
        { Text = "I'm at home. What time are you coming to dinner?"; Spam = "" }
    ]

    // make the predictions
    printfn "Модель предсказаний:"
    let predictions = messages |> List.iter(fun m -> 
            let p = engine.Predict m
            printfn "  %f %s" p.Probability m.Text)

    0 // return value

