using Random
using CSV
using DataFrames
using Statistics
using LinearAlgebra

# Load CSV data
# Function to load CSV file
function load_csv(filename)
    return CSV.File(filename) |> DataFrame
end

# Preprocess data
function preprocess_data(df)
    # Remove rows with missing values
    dropmissing!(df)

    # Convert target column to binary (assumes target is the last column)
    df[!, end] .= (df[!, end] .== ">50K") .+ 0

    # Define categorical columns
    categorical_columns = [2, 4, 6, 7, 8, 9, 10, 14]
    
    # Convert categorical columns to numerical indices
    for col in categorical_columns
        unique_values = unique(df[!, col])
        value_to_index = Dict(value => i for (i, value) in enumerate(sort(unique_values)))
        df[!, col] .= [value_to_index[val] for val in df[!, col]]
    end

    # Convert categorical columns to integers
    for col in 1:ncol(df)-1
        if col in categorical_columns
            df[!, col] .= Int.(df[!, col])
        else
            df[!, col] .= Float64.(df[!, col])
        end
    end

    return df
end

function standardize_data(df)
    for col in 1:ncol(df)-1
        column_data = df[!, col]

        mean_val = mean(column_data)
        std_val = std(column_data)
       
        df[!, col] .= (column_data .- mean_val) ./ std_val
    end

    return df
end

function train_test_split_manual(X, y, test_size=0.2)
    data = hcat(X, y)
    shuffle!(data)
    split_idx = Int(floor(size(data, 1) * (1 - test_size)))
    train_data = data[1:split_idx, :]
    test_data = data[split_idx+1:end, :]
    return train_data[:, 1:end-1], train_data[:, end], test_data[:, 1:end-1], test_data[:, end]
end

mutable struct LogisticRegression
    learning_rate::Float64
    epochs::Int64
    weights::Vector{Float64}
    bias::Float64
end

# Sigmoid function
sigmoid(z) = 1.0 ./ (1.0 .+ exp.(-z))


# Fit method to train the logistic regression model
function fit!(model::LogisticRegression, X, y)
    m, n = size(X)
    weights = model.weights
    bias = model.bias
    
    for epoch in 1:model.epochs
        predictions = sigmoid(X * weights .+ bias)
        error = predictions .- y
        
        gradient_w = (1 / m) * X' * error
        gradient_b = (1 / m) * sum(error)
        
        # Update weights and bias using gradient descent
        model.weights .-= model.learning_rate * gradient_w
        model.bias -= model.learning_rate * gradient_b

    end
end

# Predict method to make predictions
function predict(model::LogisticRegression, X, threshold=0.5)
    predictions = sigmoid(X * model.weights .+ model.bias) .>= threshold
    return predictions
end


preprocess_start = time()
# Load and preprocess data
data = load_csv("data/adult.csv")
data = preprocess_data(data)

X = data[:, 1:end-1]  
y = data[:, end]

X_train, y_train, X_test, y_test = train_test_split_manual(X, y)
X_train = standardize_data(X_train)
X_test= standardize_data(X_test)

# Convert DataFrame to Matrix (Fix: Ensure compatibility with matrix operations)
X_train = Matrix{Float64}(X_train)
X_test = Matrix{Float64}(X_test)

y_train = Vector{Float64}(y_train)
y_test = Vector{Float64}(y_test)

preprocess_end = time()  # End timing preprocessing
println("Preprocessing Time: ", preprocess_end - preprocess_start, " seconds")

# Add bias column (intercept term) to X_train and X_test
X_train = hcat(ones(size(X_train, 1)), X_train)
X_test = hcat(ones(size(X_test, 1)), X_test)

# Initialize weights
weights = randn(size(X_train, 2))

# Start timing training
train_start = time()

# Train the logistic regression model 
model = LogisticRegression(0.01, 1000, randn(size(X_train, 2)), 0.0)
fit!(model, X_train, y_train)

train_end = time()  # End timing training
println("Training Time: ", train_end - train_start, " seconds")


# Make predictions
y_pred = predict(model, X_test)

# Compute accuracy
accuracy = sum(y_pred .== y_test) / length(y_test)
println("Test Accuracy: ", accuracy)
