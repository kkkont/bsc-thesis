import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def preprocess(data):

    # Identify numerical columns excluding the target column
    numerical_cols = data.drop(columns=['Class']).select_dtypes(include=['int64', 'float64']).columns

    # Scale numerical columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data


def main():
    # Load dataset
    data = pd.read_csv('data/creditcard_2023.csv')
    target_col = 'Class'

    # Preprocess data
    data = preprocess(data)

    # Split features and target
    X = data.drop(columns=[target_col])  # Features
    y = data[target_col]  # Target variable
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    clf = RandomForestClassifier(n_estimators=2)
    clf = clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

