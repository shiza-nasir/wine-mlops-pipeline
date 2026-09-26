from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data():
    wine = load_wine(as_frame=True)

    X = wine.data
    y = wine.target

    if X.isnull().any().any():
        raise ValueError("Input features contain null values")

    if y.isnull().any():
        raise ValueError("Target contains null values")

    if X.shape[1] != 13:
        raise ValueError("Dataset must contain exactly 13 features")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test
