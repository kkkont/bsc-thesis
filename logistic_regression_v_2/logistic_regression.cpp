#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/core/data/split_data.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

std::tuple<mat, Row<size_t>> preprocess(mat dataset) {
    // Shuffle the dataset using armadillo's shuffle function
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
    // Set the seed using std::random_device
    std::random_device rd;
    int seed = rd();
    arma_rng::set_seed(seed);

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

    // Defining optimizer as MLPack's only way to set max iterations is through the optimizer and without max_iterations our model will not converge
    ens::Adam optimizer(0.001, 32); // Default learning rate and batch size
    optimizer.MaxIterations() = 100 * dataset.n_cols; // Defining the number of iterations, choosing 100 because python and Julia have this as default
    
    // Train the logistic regression model
    regression::LogisticRegression lr; 
    lr.Train(trainData, trainLabels, optimizer);   
    
    // Predict the labels for the test data
    Row<size_t> predictions;
    lr.Classify(testData, predictions);

    // Compute accuracy
    size_t correct = arma::accu(predictions == testLabels);
    double accuracy = (double) correct / testLabels.n_elem ;

    cout << "Random seed: " << seed << endl;
    cout << "Accuracy: " << accuracy << endl;
}