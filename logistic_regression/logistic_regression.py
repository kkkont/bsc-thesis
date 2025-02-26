import random
import math
import time

# Load the datased from csv file
def load_csv(filename):
    with open(filename, 'r') as file:
        data = [line.strip().split(',') for line in file]
    return data

def preprocess_data(data):
    header = data[0]
    data = data[1:]

    # Remove rows with missing values
    data = [row for row in data if '?' not in row]

    # Convert target column to binary
    for row in data:
        row[-1] = 1 if row[-1] == '>50K' else 0

    # Convert categorical columns to numerical indices manually
    categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    category_mappings = {}
    for col in categorical_columns:
        unique_values = sorted(set(row[col] for row in data))
        category_mappings[col] = {value: i for i, value in enumerate(unique_values)}
        for row in data:
            row[col] = category_mappings[col][row[col]]
    
    # Convert numerical columns to float
    for i in range(len(data)):
        data[i] = [float(val) if idx not in categorical_columns else int(val) for idx, val in enumerate(data[i])]

    return data


# Standardize features manually
def standardize_data(X):
    means = [sum(col) / len(col) for col in zip(*X)]
    stds = [math.sqrt(sum((x - mean) ** 2 for x in col) / len(col)) for col, mean in zip(zip(*X), means)]
    return [[(x - mean) / std if std != 0 else 0 for x, mean, std in zip(row, means, stds)] for row in X]

# Train-test split manually
def train_test_split_manual(data, test_size=0.2):
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]

# Sigmoid function
def sigmoid(z):
    if z < -700:  # Prevent overflow
        return 0
    elif z > 700:  # Prevent overflow
        return 1
    return 1 / (1 + math.exp(-z))



# Logistic regression class
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0
    
    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0] * n_features
        
        for _ in range(self.epochs):
            for i in range(n_samples):
                linear_model = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                prediction = sigmoid(linear_model)
                error = prediction - y[i]
                
                for j in range(n_features):
                    self.weights[j] -= self.learning_rate * error * X[i][j]
                self.bias -= self.learning_rate * error
    
    def predict(self, X):
        predictions = []
        for row in X:
            linear_model = sum(w * x for w, x in zip(self.weights, row)) + self.bias
            predictions.append(1 if sigmoid(linear_model) > 0.5 else 0)
        return predictions
    
# Track time and memory usage
start_time = time.time()

# Load, preprocess, and split the data
data = load_csv("data/adult.csv")
data = preprocess_data(data)
# Limit data to reduce training time right now. 
# data = data[:10000] 
data = standardize_data(data)
print(data[0])

X = [row[:-1] for row in data]
y = [row[-1] for row in data]

train_data, test_data = train_test_split_manual(data)
X_train, y_train = [row[:-1] for row in train_data], [row[-1] for row in train_data]
X_test, y_test = [row[:-1] for row in test_data], [row[-1] for row in test_data]

X_train = standardize_data(X_train)
X_test = standardize_data(X_test)


# Track time after preprocessing
preprocessing_time = time.time() - start_time

# Start training
model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

# Track time after training
training_time = time.time() - start_time

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = sum(1 for pred, actual in zip(y_pred, y_test) if pred == actual) / len(y_test)

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"Preprocessing Time: {preprocessing_time:.4f} seconds")
print(f"Training Time: {training_time - preprocessing_time:.4f} seconds")
