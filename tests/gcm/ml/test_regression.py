import numpy as np
from _pytest.python_api import approx
from flaky import flaky

from dowhy.gcm.ml.regression import create_polynom_regressor


@flaky(max_runs=3)
def test_when_fit_and_predict_polynom_regressor_then_returns_accurate_results():
    X = np.random.normal(0, 1, (100, 2))
    Y = X[:, 0] * X[:, 1]

    mdl = create_polynom_regressor(degree=2)
    mdl.fit(X, Y)

    X_test = np.random.normal(0, 1, (100, 2))
    Y_test = X_test[:, 0] * X_test[:, 1]

    assert mdl.predict(X_test).reshape(-1) == approx(Y_test, abs=1e-10)


@flaky(max_runs=3)
def test_when_given_categorical_training_data_when_fit_and_predict_polynom_regressor_then_returns_accurate_results():
    def _generate_data():
        X = np.column_stack(
            [np.random.choice(2, 100, replace=True).astype(str), np.random.normal(0, 1, (100, 2)).astype(object)]
        ).astype(object)
        Y = []
        for i in range(X.shape[0]):
            Y.append(X[i, 1] * X[i, 2] if X[i, 0] == "0" else X[i, 1] + X[i, 2])

        return X, np.array(Y)

    X_training, Y_training = _generate_data()
    X_test, Y_test = _generate_data()
    mdl = create_polynom_regressor(degree=3)
    mdl.fit(X_training, Y_training)

    assert mdl.predict(X_test).reshape(-1) == approx(Y_test, abs=1e-10)
