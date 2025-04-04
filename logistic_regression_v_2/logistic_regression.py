import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def preprocess(data):

    # Identify categorical and numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns[:-1]

    # Scale numerical columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data


def main():
    randint = random.randint(0, 1000)  # Set random seed for reproducibility
    # Load dataset
    data = pd.read_csv('../data/creditcard_2023.csv')
    target_col = 'Class'

    # Preprocess data
    data = preprocess(data)

    # Split features and target
    X = data.drop(columns=[target_col])  # Features
    y = data[target_col]  # Target variable
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randint)

    # Load and train the model
    model = LogisticRegression(random_state=randint)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Seed: {randint}")
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()

