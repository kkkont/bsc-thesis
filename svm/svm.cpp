#include <mlpack/core.hpp>
#include <mlpack/methods/linear_svm/linear_svm.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/core/data/split_data.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

std::tuple<mat, Row<size_t>> preprocess(mat dataset) {
    // Shuffle the dataset using armadillo's shuffle function
    arma_rng::set_seed_random();
    uvec indices = shuffle(regspace<uvec>(0, dataset.n_cols - 1));
    dataset = dataset.cols(indices);

    // Separate features and labels
    Row<size_t> labels = conv_to<Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
    mat features = dataset.rows(0, dataset.n_rows - 2);
   
    // Standardize the features using MLPack's StandardScaler
    data::StandardScaler scaler;
    mat features_scaled;
    scaler.Fit(features);
    scaler.Transform(features, features_scaled);

    // Return the scaled features and labels
    return std::make_tuple(features_scaled, labels);
}

int main() {
    // Read the CSV file
    mat dataset;
    data::Load("../data/creditcard_2023_without_header.csv", dataset, true);

    // Call the preprocessing function
    mat X;
    Row<size_t> y;
    std::tie(X, y) = preprocess(dataset);
   
    // Split the data into training and test sets
    mat trainData, testData;
    Row<size_t> trainLabels, testLabels;
    data::Split(X, y, trainData, testData, trainLabels, testLabels, 0.20);

    // Train the model
    svm::LinearSVM<> svm;   
    // Train the model for 2 classes
    svm.Train(trainData, trainLabels, 2);   
    
    // Predict the labels for the test data
    Row<size_t> predictions;
    svm.Classify(testData, predictions);

    // Compute accuracy
    size_t correct = arma::accu(predictions == testLabels);
    double accuracy = (double) correct / testLabels.n_elem * 100.0;

    cout << "Accuracy on test set: " << accuracy << "%" << endl;
}