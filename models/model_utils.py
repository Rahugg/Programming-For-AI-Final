import os
import json
import numpy as np
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def _silence_lib_warnings():
    """Silence noisy runtime and convergence warnings emitted during fit/transform/predict."""
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.decomposition\._base")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.linear_model\._base")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"xgboost")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- опциональный XGBoost, не падаем если не установлен
try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False


def train_models(X_train, y_train):
    """
    Набор моделей:
      - Dummy (baseline)
      - HuberRegressor (устойчивый линейный) + RobustScaler
      - RandomForest
      - SVR + RobustScaler
      - (опционально) XGBoost
    """
    models = {
        "Dummy(mean)": DummyRegressor(strategy="mean"),
        "Huber": make_pipeline(
            RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0)),
            HuberRegressor(epsilon=1.35, alpha=0.0001, fit_intercept=True, max_iter=1000)
        ),
        "RandomForest": RandomForestRegressor(random_state=42),
        "SVR": make_pipeline(
            RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0)),
            SVR(C=10.0, epsilon=0.1, kernel="rbf")
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(objective="reg:squarederror", random_state=42)

    trained = {}
    # приглушим именно эти «шумные» RuntimeWarning из extmath, не скрывая других предупреждений
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained[name] = model
    return trained


def hyperparameter_tuning(model, param_grid, X_train, y_train):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=2
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
        gs.fit(X_train, y_train)
    return gs.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, {"MSE": float(mse), "RMSE": rmse, "MAE": float(mae), "R2": float(r2)}


def persist_model(model, columns_template, path_dir="models", fname="best_model.joblib"):
    os.makedirs(path_dir, exist_ok=True)
    bundle = {"model": model, "columns": columns_template}
    path = os.path.join(path_dir, fname)
    joblib.dump(bundle, path)
    return path


def load_model(path):
    bundle = joblib.load(path)
    return bundle["model"], bundle["columns"]
