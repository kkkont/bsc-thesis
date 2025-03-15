using CSV
using DataFrames
using MLJ
using MLJScikitLearnInterface
using MLJBase: machine, fit!, transform, coerce
using DataFrames: DataFrame, dropmissing

function preprocess(data::DataFrame)
    # Extract target column
    y = data.Class
    data = data[:, Not(:Class)]
  
    # Normalize numerical columns
    stand = Standardizer(count=true)
    X = transform(fit!(machine(stand, data)), data)

    # Coerce the target column to a multiclass type
    y = coerce(y, Multiclass)
    
    return X, y
end

function main()
    # Generate and print random seed
    rng = rand(1:1000)

    # Load data
    data = CSV.File("../data/creditcard_2023.csv") |> DataFrame
    X, y = preprocess(data)

    # Split the data, 80% for training and 20% for testing, shuffles the data
    (X_train, X_test), (y_train, y_test) = partition((X, y), 0.8, rng = rng, multi=true)

    # Load the model
    SVMLinearClassifier = @load SVMLinearClassifier pkg=MLJScikitLearnInterface
    model = SVMLinearClassifier(random_state=rng)

    # Train the model
    mach = machine(model, X_train, y_train)
    fit!(mach)

    # Make predictions
    ŷ = predict(mach, X_test)

    #Calculate accuracy
    accuracy = mean(ŷ.== y_test)
    println("Random Seed: ", rng)
    println("Accuracy: $accuracy")
end

main()