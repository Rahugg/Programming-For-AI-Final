from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Файл с функциями для обучения моделей и оценки
def train_models(X_train, y_train):
    """
    Сравнение минимум 3 моделей (LR, RF, SVM, XGBoost)
    """
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    """
    Подбор гиперпараметров через GridSearchCV с отображением процесса
    """
    print(f"\n=== Hyperparameter tuning для {model.__class__.__name__} ===")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=2  # <-- показываем прогресс
    )
    grid_search.fit(X_train, y_train)
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучший R² на валидации: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Вычисление ошибок: MSE, RMSE, R2
    Матрица ошибок + precision, recall, F1 для регрессии можно заменить на MAE/MSE/R2
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    return y_pred
