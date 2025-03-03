import random
import math
import time

# Load dataset
def load_csv(filename):
    with open(filename, 'r') as file:
        data = [line.strip().split(',') for line in file]
    return data[1:]  # Skip header

def preprocess_data(data):
    # Remove rows with missing values
    data = [row for row in data if '?' not in row]

    # Define column indices
    categorical_cols = [1, 3, 5, 6, 7, 8, 9, 13]
    numerical_cols = [0, 2, 4, 10, 11, 12]
    target_col = -1

    # Convert target variable to binary (1 for '>50K', 0 for '<=50K')
    for row in data:
        row[target_col] = 1 if row[target_col] == '>50K' else 0  # Binary target

    # Encode categorical columns
    mappings = {col: {val: i for i, val in enumerate(sorted(set(row[col] for row in data)))} for col in categorical_cols}
    for row in data:
        for col in categorical_cols:
            row[col] = mappings[col][row[col]]  # Replace categorical values with encoded values

    # Separate features and target
    features = [row[:-1] for row in data]  # All columns except the target
    target = [row[-1] for row in data]  # Only the target column

    # Convert all numerical columns to floats before normalization
    for row in features:
        for col in numerical_cols:
            row[col] = float(row[col])  # Convert to float to ensure numerical operations

    # Normalize numerical columns (Min-Max scaling)
    numerical_data = [[row[col] for col in numerical_cols] for row in features]
    normalized_numerical_data = normalize(numerical_data)

    # Replace the numerical columns in features with normalized data
    for i, row in enumerate(features):
        for j, col in enumerate(numerical_cols):
            row[col] = normalized_numerical_data[i][j]

    # Combine features and target back together
    final_data = [row + [target[i]] for i, row in enumerate(features)]  # Append target to the feature list

    return final_data

# Normalize features (Min-Max scaling)
def normalize(X):
    mins = [min(col) for col in zip(*X)]
    maxs = [max(col) for col in zip(*X)]
    return [[(x - mn) / (mx - mn) if mx > mn else 0 for x, mn, mx in zip(row, mins, maxs)] for row in X]

# Sigmoid function with clamping
def sigmoid(z):
    # Clamp the value to avoid overflow issues
    if z > 700:
        return 1
    elif z < -700:
        return 0
    return 1 / (1 + math.exp(-z))

# Train-test split
def train_test_split(data, test_size=0.2):
    random.shuffle(data)
    split = int(len(data) * (1 - test_size))
    return data[:split], data[split:]

# Logistic Regression model
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def fit(self, X, y):
        self.weights = [0] * len(X[0])
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sigmoid(sum(w * x for w, x in zip(self.weights, xi)) + self.bias)
                error = pred - yi
                for j in range(len(self.weights)):
                    self.weights[j] -= self.lr * error * xi[j]
                self.bias -= self.lr * error

    def predict(self, X):
        return [1 if sigmoid(sum(w * x for w, x in zip(self.weights, row)) + self.bias) > 0.5 else 0 for row in X]

# Measure time
start_time = time.time()

# Load & preprocess data
data = load_csv("data/adult.csv")

# Track preprocessing time
preprocess_start = time.time()
data = preprocess_data(data)
X, y = [row[:-1] for row in data], [row[-1] for row in data]
preprocess_time = time.time() - preprocess_start

# Train-test split
train, test = train_test_split(data)
X_train, y_train = [row[:-1] for row in train], [row[-1] for row in train]
X_test, y_test = [row[:-1] for row in test], [row[-1] for row in test]
print(len(X_train))
# Track training time
train_start = time.time()
model = LogisticRegression(lr=0.01, epochs=500)
model.fit(X_train, y_train)
train_time = time.time() - train_start

# Evaluate model
y_pred = model.predict(X_test)
accuracy = sum(1 for p, a in zip(y_pred, y_test) if p == a) / len(y_test)

# Results
total_time = time.time() - start_time
print(f"Accuracy: {accuracy:.4f}")
print(f"Preprocessing Time: {preprocess_time:.4f} seconds")
print(f"Training Time: {train_time:.4f} seconds")
print(f"Total Time: {total_time:.4f} seconds")
