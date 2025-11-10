import os
import json
import numpy as np
import pandas as pd

from utils.data_utils import load_data, preprocess_data, split_data
from models.model_utils import (
    train_models,
    hyperparameter_tuning,
    evaluate_model,
    persist_model,
    load_model,
)
import warnings
from sklearn.exceptions import ConvergenceWarning
from utils.feature_selection import select_k_best, rfe_selection, apply_pca
from report.error_analysis import error_summary, save_error_report
from ui.ui import make_feature_row

from sklearn.ensemble import RandomForestRegressor


# ---------- конфигурация путей ----------

def _find_dataset():
    """
    Ищет первый существующий CSV из списка кандидатов рядом со скриптом
    или в рабочей директории.
    """
    candidates = [
        "data/cardetailsv4.csv",
        "data/cardetailsv3.csv",
        "data/cardata.csv",
        "data/cardetailsfromcardecho.csv",
    ]
    here = os.path.dirname(os.path.abspath(__file__))
    # 1) рядом со скриптом
    for name in candidates:
        p = os.path.join(here, name)
        if os.path.exists(p):
            return p
    # 2) в текущей рабочей директории
    for name in candidates:
        p = os.path.join(os.getcwd(), name)
        if os.path.exists(p):
            return p
    # по умолчанию — рядом со скриптом (можно заменить вручную)
    return os.path.join(here, "cardetailsv4.csv")


DATA_PATH = _find_dataset()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.decomposition\._base")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.linear_model\._base")
warnings.filterwarnings("ignore", category=UserWarning, module=r"xgboost")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------- сетка гиперпараметров для RF ----------

param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}


# ---------- основной обучающий сценарий ----------

def train_compare_with_feature_selection():
    print("\n[1/4] Загрузка и подготовка данных...")
    print(f"DATA_PATH: {DATA_PATH}")
    df = load_data(DATA_PATH)
    X, y, columns_template = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("[2/4] Подбор гиперпараметров для RandomForest (GridSearchCV, scoring=R2)...")
    best_rf = hyperparameter_tuning(
        RandomForestRegressor(random_state=42), param_grid_rf, X_train, y_train
    )

    print("[3/4] Обучение семейства базовых моделей (включая Baseline)...")
    base_models = train_models(X_train, y_train)  # Dummy/RidgeCV/RF/SVR/(XGB если есть)
    # Добавим тюнингованный RF под отдельным именем
    base_models["Best RandomForest"] = best_rf

    # === ветки отбора признаков (обучаем отдельные RF на трансформированных данных) ===
    print("[3.1] SelectKBest...")
    Xk_train, selector_k, kbest_names = select_k_best(
        X_train, y_train, k=min(10, X_train.shape[1])
    )
    Xk_test = selector_k.transform(X_test)
    rf_kbest = RandomForestRegressor(random_state=42).fit(Xk_train, y_train)

    print("[3.2] RFE...")
    rfe_model = RandomForestRegressor(random_state=42)
    Xr_train, selector_r, rfe_names = rfe_selection(
        rfe_model, X_train, y_train, n_features=min(10, X_train.shape[1])
    )
    Xr_test = selector_r.transform(X_test)
    rf_rfe = RandomForestRegressor(random_state=42).fit(Xr_train, y_train)

    print("[3.3] PCA (со стандартизацией внутри пайплайна)...")
    Xp_train, pca_pipe = apply_pca(X_train, n_components=min(10, X_train.shape[1]))
    Xp_test = pca_pipe.transform(X_test)
    rf_pca = RandomForestRegressor(random_state=42).fit(Xp_train, y_train)

    # Соберём все модели под единый словарь для сравнения метрик
    all_models = {}
    all_models.update(base_models)
    all_models["RF + SelectKBest"] = rf_kbest
    all_models["RF + RFE"] = rf_rfe
    all_models["RF + PCA"] = rf_pca

    print("\n[4/4] Оценка моделей...")
    metrics_map = {}
    # чтобы корректно подавать соответствующие тестовые матрицы для веток селекции
    for name, model in all_models.items():
        if name == "RF + SelectKBest":
            y_pred, metrics = evaluate_model(model, Xk_test, y_test)
        elif name == "RF + RFE":
            y_pred, metrics = evaluate_model(model, Xr_test, y_test)
        elif name == "RF + PCA":
            y_pred, metrics = evaluate_model(model, Xp_test, y_test)
        else:
            y_pred, metrics = evaluate_model(model, X_test, y_test)

        metrics_map[name] = metrics
        print(f"\n{name}: " + json.dumps(metrics, ensure_ascii=False))

    # Лучшая модель среди всех (для отчёта)
    best_overall_name = max(metrics_map, key=lambda k: metrics_map[k]["R2"])

    # Лучшая «регулярная» модель (работает в пространстве исходных признаков) — её СНИМАЕМ и сохраняем
    regular_names = [n for n in all_models.keys()
                     if n not in {"RF + SelectKBest", "RF + RFE", "RF + PCA"}]
    best_regular_name = max(regular_names, key=lambda k: metrics_map[k]["R2"])
    best_regular_model = all_models[best_regular_name]

    # Сохраняем ТОЛЬКО «регулярную» модель + шаблон колонок исходного пространства
    model_path = persist_model(best_regular_model, list(X.columns), path_dir=MODELS_DIR)

    # Сохраняем метрики и информацию о селекции
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics_map,
                "best_overall": best_overall_name,
                "best_regular": best_regular_name,
                "model_path": model_path,
                "kbest_features": kbest_names,
                "rfe_features": rfe_names,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Анализ ошибок — по сохранённой «регулярной» модели в исходном пространстве
    y_pred_best, _ = evaluate_model(best_regular_model, X_test, y_test)
    full_df, top_df = error_summary(y_test, y_pred_best, top_n=15)
    full_path, top_path = save_error_report(
        full_df, top_df, out_dir=REPORTS_DIR, base_name="best_model"
    )

    print(f"\nСохранены отчёты об ошибках:\n - {full_path}\n - {top_path}")
    print(f"\nЛучшая среди всех: {best_overall_name}")
    print(f"Сохранена регулярная лучшая: {best_regular_name} → {model_path}")
    print(f"\nВыбранные признаки (SelectKBest): {kbest_names}")
    print(f"Выбранные признаки (RFE): {rfe_names}")


# ---------- меню: предсказание ----------

def predict_menu():
    if not os.path.exists(BEST_MODEL_PATH):
        print("Сначала обучите модель (меню пункт 1).")
        return
    model, columns = load_model(BEST_MODEL_PATH)
    print("\nВведите параметры автомобиля (Enter — взять значение по умолчанию):")
    row = make_feature_row(columns)
    pred = float(model.predict(row)[0])
    print(f"\nОценка стоимости: {pred:,.2f}")


# ---------- меню: метрики и отчёты ----------

def error_report_menu():
    if not os.path.exists(METRICS_PATH):
        print("Сначала обучите модель (меню пункт 1).")
        return
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    print("\n=== Метрики моделей ===")
    for name, m in report["metrics"].items():
        print(f"- {name}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.1f}, MAE={m['MAE']:.1f}")

    print(f"\nЛучшая среди всех: {report['best_overall']}")
    print(f"Сохранённая регулярная лучшая: {report['best_regular']} ({report['model_path']})")
    print(f"SelectKBest топ-фичи: {report.get('kbest_features')}")
    print(f"RFE топ-фичи: {report.get('rfe_features')}")
    print("\nФайлы с ошибками см. в папке reports/: best_model_full.csv, best_model_top.csv")


# ---------- главное меню ----------

def main_menu():
    while True:
        print("\n=== Car Price Prediction ===")
        print("1 - Train/compare models + Feature Selection + Error Analysis")
        print("2 - Predict car price (интерактивный ввод)")
        print("3 - Show metrics & error reports info")
        print("4 - Exit")
        choice = input("Выберите опцию: ").strip()

        if choice == "1":
            try:
                train_compare_with_feature_selection()
            except Exception as e:
                print(f"Ошибка при обучении: {e}")
        elif choice == "2":
            try:
                predict_menu()
            except Exception as e:
                print(f"Ошибка при предсказании: {e}")
        elif choice == "3":
            try:
                error_report_menu()
            except Exception as e:
                print(f"Ошибка при чтении отчётов: {e}")
        elif choice == "4":
            print("Выход.")
            break
        else:
            print("Неверная опция, попробуйте снова.")


if __name__ == "__main__":
    main_menu()
