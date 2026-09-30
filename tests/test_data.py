from src.data import load_data


def test_feature_count():
    X_train, X_test, y_train, y_test = load_data()

    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13


def test_no_null_values():
    X_train, X_test, y_train, y_test = load_data()

    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    assert y_train.isnull().sum() == 0
    assert y_test.isnull().sum() == 0


def test_train_test_split():
    X_train, X_test, y_train, y_test = load_data()

    assert len(X_train) == 142
    assert len(X_test) == 36
    assert len(y_train) == 142
    assert len(y_test) == 36


def test_all_classes_present():
    X_train, X_test, y_train, y_test = load_data()

    assert set(y_train.unique()) == {0, 1, 2}
    assert set(y_test.unique()) == {0, 1, 2}
