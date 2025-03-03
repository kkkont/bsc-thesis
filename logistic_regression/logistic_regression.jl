using Random
using DataFrames
using Statistics
using LinearAlgebra
using CSV

function preprocess_data(data::DataFrame)
    for col in names(data)
        if eltype(data[!, col]) <: AbstractString  # Ensure it's a string column
            data[!, col] = replace(data[!, col], "?" => missing)
        end
    end
    
    # Drop rows with missing values
    data = dropmissing(data) 

    # Ensure categorical columns are stored as standard String arrays before encoding
    categorical_cols = [2, 4, 6, 7, 8, 9, 10, 14]  # 1-based indexing in Julia
    numerical_cols = [1, 3, 5, 11, 12, 13]
    target_col = ncol(data)  # Target column is the last column

    # Convert target variable to binary (1 for '>50K', 0 for '<=50K')
    data[!, target_col] .= (data[:, target_col] .== ">50K") .+ 0  # Binary target

    # Convert categorical columns to standard String vectors
    for col in categorical_cols
        data[!, col] = string.(data[:, col])  # Convert to standard String array
    end

    # Encode categorical columns
    mappings = Dict{Int, Dict{String, Int}}()
    for col in categorical_cols
        unique_vals = sort(unique(data[:, col]))
        mappings[col] = Dict(val => i for (i, val) in enumerate(unique_vals))
    end

    # Replace categorical values with encoded values
    for col in categorical_cols
        data[!, col] .= [mappings[col][val] for val in data[:, col]]
    end

    # Convert all numerical columns to Float64 before normalization
    for col in numerical_cols
        data[!, col] .= float.(data[:, col])
    end

    # Normalize numerical columns (Min-Max scaling)
    numerical_data = Matrix(data[:, numerical_cols])  # Extract numerical data as a matrix
    normalized_numerical_data = normalize(numerical_data)

    # Replace numerical columns with normalized values
    data[:, numerical_cols] .= normalized_numerical_data

    return data
end

# Normalize features (Min-Max scaling)
function normalize(X::Matrix)
    mins = minimum(X, dims=1)
    maxs = maximum(X, dims=1)
    return (X .- mins) ./ (maxs .- mins)  # Element-wise Min-Max scaling
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
data = CSV.read("data/adult.csv", DataFrame) 
data = preprocess_data(data)

X = data[:, 1:end-1]  
y = data[:, end]

X_train, y_train, X_test, y_test = train_test_split_manual(X, y)


println(X_train[1:5,:])
println(size(X_train))
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
