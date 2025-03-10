#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/core/data/split_data.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

// Function to read the CSV file
vector<vector<string>> readCSV(const string& filename) {
    ifstream file(filename);
    string line;
    vector<vector<string>> data;

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<string> row;

        while (getline(ss, value, ',')) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    return data;
}

// Function to label encode categorical columns manually as with MLPack it does not read in the categorical data correctly or in format we want.
void labelEncode(vector<vector<string>>& data) {
    vector<int> categoricalColumns = {1,3,5,6,7,8,9,13,14}; 
    // Map for encoding
    for (int col : categoricalColumns) {
        unordered_map<string, int> labelMap;
        int label = 0;

        // Loop over rows and create label encoding for each category
        for (auto& row : data) {
            string category = row[col];
            if (labelMap.find(category) == labelMap.end()) {
                labelMap[category] = label++;
            }
            row[col] = to_string(labelMap[category]);  // Store the label as a string
        }
    }
}

// Function to convert data to Armadillo matrix
mat convertToMatrix(const vector<vector<string>>& data) {
    int numRows = data.size();
    int numCols = data[0].size();
    mat matrix(numRows, numCols);

    // Convert data to matrix
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            matrix(i, j) = stod(data[i][j]);
        }
    }
    return matrix;
}

mat preprocess(const string& filename){
    vector<vector<string>> data = readCSV(filename);

    // Remove rows with missing values
    data.erase(remove_if(data.begin(), data.end(), [](const vector<string>& row) {
        return any_of(row.begin(), row.end(), [](const string& value) { return value == "?"; });
    }), data.end());

    labelEncode(data);
    mat datasetMatrix = convertToMatrix(data);
    return datasetMatrix;
}

int main() {
    // Read the CSV file
    string filename = "../data/adult_without_header.csv";

    mat data = preprocess(filename);
   
    mat X = data.cols(0, data.n_cols - 2).t(); // Transpose to match dimensions
    Row<size_t> y = conv_to<arma::Row<size_t>>::from(data.col(data.n_cols - 1));

    data::StandardScaler scaler;
    mat X_scaled;
    scaler.Fit(X);
    scaler.Transform(X, X_scaled);


    mat trainData, testData;
    Row<size_t> trainLabels, testLabels;
    data::Split(X_scaled, y, trainData, testData, trainLabels, testLabels, 0.2, true);

    // Train the logistic regression model
    regression::LogisticRegression<> lr; // Step 1: create model.
    lr.Train(trainData, trainLabels);   // Step 2: train model.
    
    // Compute accuracy of test data 
    cout << "Accuracy on test set: "
    << lr.ComputeAccuracy(testData, testLabels) << "%" << endl;
}