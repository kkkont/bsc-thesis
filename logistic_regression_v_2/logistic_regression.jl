using CSV
using DataFrames
using MLJ
using MLJLinearModels
using MLJBase: machine, fit!, transform, coerce
using DataFrames: DataFrame, dropmissing

function preprocess(data::DataFrame)
    # Drop missing values, disallowmissing=true ensures that the whole row is deleted
    data = dropmissing(data, disallowmissing=true)

    # Perform label encoding for categorical variables.
    data = coerce(data, :workclass => Multiclass,
                  :education => Multiclass,
                  Symbol("marital-status") => Multiclass,
                  :occupation => Multiclass,
                  :relationship => Multiclass,
                  :race => Multiclass,
                  :gender => Multiclass,
                  Symbol("native-country") => Multiclass,
                  :income => OrderedFactor)
    
    y = data.income
    data = data[:, Not(:income)]
    # One-hot encode categorical variables, excluding the target column :income
    hot = OneHotEncoder()
    data = transform(fit!(machine(hot, data)), data)
    
    # Normalize numerical columns
    stand = Standardizer(count=true)
    X = transform(fit!(machine(stand, data)), data)
    
    return X, y
end

function main()
    # Load data
    data = CSV.File("data/adult.csv", missingstring=["?"]) |> DataFrame
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