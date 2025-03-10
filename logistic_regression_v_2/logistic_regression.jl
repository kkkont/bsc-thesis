using CSV
using DataFrames
using MLJ
using MLJLinearModels
using MLJBase: machine, fit!, transform, coerce
using DataFrames: DataFrame, dropmissing

function preprocess(data::DataFrame)
   
    y = data.Class
    data = data[:, Not(:Class)]
  
    # Normalize numerical columns
    stand = Standardizer(count=true)
    X = transform(fit!(machine(stand, data)), data)

    y = coerce(y, Multiclass)
    
    return X, y
end

function main()
    # Load data
    data = CSV.File("data/creditcard_2023.csv") |> DataFrame
    X, y = preprocess(data)

    # Split the data, 80% for training and 20% for testing, shuffles the data
    (X_train, X_test), (y_train, y_test) = partition((X, y), 0.8, shuffle=true, multi=true)

    LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels

    model = LogisticClassifier()

    mach = machine(model, X_train, y_train)
    fit!(mach)

    ŷ = predict(mach, X_test)
    ŷ_labels = mode.(ŷ) 

    accuracy = mean(ŷ_labels .== y_test)
    println("Accuracy: $accuracy")
end

main()