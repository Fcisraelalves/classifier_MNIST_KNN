import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def load_split_mnist(train_csv_path : str, test_csv_path : str):
    train_data = pd.read_csv(train_csv_path).to_numpy()
    test_data = pd.read_csv(test_csv_path).to_numpy()

    X_train = train_data[:, 1:]
    X_test = test_data[:, 1:]
    y_train = train_data[:, 0]
    y_test = test_data[:, 0]

    return X_train, X_test, y_train, y_test

def save_model(path_to_save : str, model : KNeighborsClassifier | StandardScaler) -> None:
    joblib.dump(model, path_to_save)

def load_model(model_path : str) -> KNeighborsClassifier | StandardScaler:
    model = joblib.load(model_path)
    return model
    