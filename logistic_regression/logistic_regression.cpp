#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <ctime>
#include <unordered_set>
#include <cmath>
#include <random>

// Function to read CSV file and return data as a vector of vectors (rows)
std::vector<std::vector<std::string>> load_csv(const std::string &filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    std::string line;

    // Ignore the first line (header)
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            // Trim whitespace from the value
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
            value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), value.end());
            row.push_back(value);
        }
        data.push_back(row);
    }
    return data;
}

// Function to normalize data using Min-Max scaling
std::vector<std::vector<float>> normalize(const std::vector<std::vector<float>> &X, const std::vector<int> &numerical_cols) {
    std::vector<float> mins(numerical_cols.size(), std::numeric_limits<float>::max());
    std::vector<float> maxs(numerical_cols.size(), std::numeric_limits<float>::lowest());
    
    // Find the min and max for each numerical column
    for (const auto &row : X) {
        for (size_t i = 0; i < numerical_cols.size(); ++i) {
            int col = numerical_cols[i];
            mins[i] = std::min(mins[i], row[col]);
            maxs[i] = std::max(maxs[i], row[col]);
        }
    }

    // Normalize the data
    std::vector<std::vector<float>> normalized_data;
    for (const auto &row : X) {
        std::vector<float> normalized_row;
        for (size_t i = 0; i < numerical_cols.size(); ++i) {
            int col = numerical_cols[i];
            float normalized_value = (maxs[i] > mins[i]) ? (row[col] - mins[i]) / (maxs[i] - mins[i]) : 0;
            normalized_row.push_back(normalized_value);
        }
        normalized_data.push_back(normalized_row);
    }
    
    return normalized_data;
}

// Function to preprocess the data
std::vector<std::vector<std::string>> preprocess_data(std::vector<std::vector<std::string>> &data) {
    // Remove rows with missing values (represented by '?')
    data.erase(std::remove_if(data.begin(), data.end(), [](const std::vector<std::string> &row) {
        return std::find(row.begin(), row.end(), "?") != row.end();
    }), data.end());

    // Define column indices
    std::vector<int> categorical_cols = {1, 3, 5, 6, 7, 8, 9, 13};
    std::vector<int> numerical_cols = {0, 2, 4, 10, 11, 12};
    int target_col = data[0].size() - 1; // Assuming last column is the target
    
    // Convert target variable to binary (1 for '>50K', 0 for '<=50K')
    for (auto &row : data) {
        row[target_col] = (row[target_col] == ">50K") ? "1" : "0";
    }

    // Encode categorical columns
    std::unordered_map<int, std::unordered_map<std::string, int>> mappings;
    for (int col : categorical_cols) {
        std::unordered_set<std::string> unique_values;
        for (const auto &row : data) {
            unique_values.insert(row[col]);
        }
        
        int index = 0;
        for (const auto &value : unique_values) {
            mappings[col][value] = index++;
        }
    }
    
    for (auto &row : data) {
        for (int col : categorical_cols) {
            row[col] = std::to_string(mappings[col][row[col]]);
        }
    }

    // Separate features and target
    std::vector<std::vector<float>> features;
    std::vector<int> target;
    for (const auto &row : data) {
        std::vector<float> feature_row;
        for (int i = 0; i < row.size() - 1; ++i) {
            feature_row.push_back(std::stof(row[i])); // Convert to float
        }
        features.push_back(feature_row);
        target.push_back(std::stoi(row[target_col])); // Target is binary
    }

    // Normalize numerical columns (Min-Max scaling)
    std::vector<std::vector<float>> normalized_data = normalize(features, numerical_cols);

    // Replace the numerical columns with normalized data
    for (size_t i = 0; i < features.size(); ++i) {
        for (size_t j = 0; j < numerical_cols.size(); ++j) {
            features[i][numerical_cols[j]] = normalized_data[i][j];
        }
    }

    // Combine features and target back together
    std::vector<std::vector<std::string>> final_data;
    for (size_t i = 0; i < features.size(); ++i) {
        std::vector<std::string> row;
        for (const auto &value : features[i]) {
            row.push_back(std::to_string(value));
        }
        row.push_back(std::to_string(target[i]));  // Add target
        final_data.push_back(row);
    }

    return final_data;
}

// Function to split data into training and testing sets
void train_test_split(const std::vector<std::vector<std::string>> &data, std::vector<std::vector<std::string>> &train_data, std::vector<std::vector<std::string>> &test_data, float test_size = 0.2) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<std::vector<std::string>> shuffled_data = data;
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), g);

    size_t test_count = static_cast<size_t>(test_size * data.size());
    test_data.assign(shuffled_data.begin(), shuffled_data.begin() + test_count);
    train_data.assign(shuffled_data.begin() + test_count, shuffled_data.end());
}

// Sigmoid function
float sigmoid(float z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// Function to train logistic regression model
std::vector<float> train_logistic_regression(const std::vector<std::vector<float>> &X, const std::vector<int> &y, float learning_rate = 0.01, int epochs = 1000) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    std::vector<float> weights(n_features, 0.0);
    float bias = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<float> linear_model(n_samples, 0.0);
        std::vector<float> y_pred(n_samples, 0.0);

        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                linear_model[i] += X[i][j] * weights[j];
            }
            linear_model[i] += bias;
            y_pred[i] = sigmoid(linear_model[i]);
        }

        std::vector<float> dw(n_features, 0.0);
        float db = 0.0;

        for (size_t i = 0; i < n_samples; ++i) {
            float error = y_pred[i] - y[i];
            for (size_t j = 0; j < n_features; ++j) {
                dw[j] += X[i][j] * error;
            }
            db += error;
        }

        for (size_t j = 0; j < n_features; ++j) {
            weights[j] -= learning_rate * dw[j] / n_samples;
        }
        bias -= learning_rate * db / n_samples;
    }

    weights.push_back(bias);
    return weights;
}

// Function to predict using logistic regression model
std::vector<int> predict_logistic_regression(const std::vector<std::vector<float>> &X, const std::vector<float> &weights) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    std::vector<int> predictions(n_samples, 0);
    float bias = weights.back();

    for (size_t i = 0; i < n_samples; ++i) {
        float linear_model = 0.0;
        for (size_t j = 0; j < n_features; ++j) {
            linear_model += X[i][j] * weights[j];
        }
        linear_model += bias;
        predictions[i] = sigmoid(linear_model) >= 0.5 ? 1 : 0;
    }

    return predictions;
}

// Function to calculate accuracy
float calculate_accuracy(const std::vector<int> &y_true, const std::vector<int> &y_pred) {
    size_t correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / y_true.size();
}

int main() {
    // Load data from CSV file
    std::vector<std::vector<std::string>> data = load_csv("../data/adult.csv");

    // Track preprocessing time
    clock_t preprocess_start = clock();
    data = preprocess_data(data);
    clock_t preprocess_end = clock();
    
    double preprocess_time = double(preprocess_end - preprocess_start) / CLOCKS_PER_SEC;
    std::cout << "Preprocessing time: " << preprocess_time << " seconds" << std::endl;

    // Split data into training and testing sets
    std::vector<std::vector<std::string>> train_data, test_data;
    train_test_split(data, train_data, test_data);

    // Separate features and target for training data
    std::vector<std::vector<float>> X_train;
    std::vector<int> y_train;
    for (const auto &row : train_data) {
        std::vector<float> feature_row;
        for (size_t i = 0; i < row.size() - 1; ++i) {
            feature_row.push_back(std::stof(row[i]));
        }
        X_train.push_back(feature_row);
        y_train.push_back(std::stoi(row.back()));
    }

    // Separate features and target for testing data
    std::vector<std::vector<float>> X_test;
    std::vector<int> y_test;
    for (const auto &row : test_data) {
        std::vector<float> feature_row;
        for (size_t i = 0; i < row.size() - 1; ++i) {
            feature_row.push_back(std::stof(row[i]));
        }
        X_test.push_back(feature_row);
        y_test.push_back(std::stoi(row.back()));
    }
 
    // Track training time
    clock_t train_start = clock();
    // Train logistic regression model
    std::vector<float> weights = train_logistic_regression(X_train, y_train);
    clock_t train_end = clock();
    
    double train_time = double(train_end - train_start) / CLOCKS_PER_SEC;
    std::cout << "Training time: " << train_time << " seconds" << std::endl;

    // Predict on test data
    std::vector<int> y_pred = predict_logistic_regression(X_test, weights);

    // Calculate accuracy
    float accuracy = calculate_accuracy(y_test, y_pred);
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}
