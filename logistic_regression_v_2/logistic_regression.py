import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def preprocess(data):
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)

    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

    # Label Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Save encoders in case we need to inverse transform later

    # Scale numerical columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data


def main():
    # Load dataset
    data = pd.read_csv('data/adult.csv')
    target_col = 'income'

    # Preprocess data
    data = preprocess(data)

    # Split features and target
    X = data.drop(columns=[target_col])  # Features
    y = data[target_col]  # Target variable

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

