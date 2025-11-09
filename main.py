import pandas as pd
from utils.data_utils import load_data, preprocess_data, split_data
from models.model_utils import train_models, hyperparameter_tuning, evaluate_model
from sklearn.ensemble import RandomForestRegressor

# Путь к данным
DATA_PATH = "data/cardetailsv4.csv"

# Параметры для GridSearch
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

def main_menu():
    print("=== Car Price Prediction ===")
    print("1 - Train and compare models")
    print("2 - Predict car price")
    print("3 - Exit")
    choice = input("Выберите опцию: ")

    if choice == "1":
        # 1. Загружаем и подготавливаем данные
        df = load_data(DATA_PATH)
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        # 2. Подбор гиперпараметров для RandomForest
        best_rf = hyperparameter_tuning(RandomForestRegressor(random_state=42),
                                        param_grid_rf, X_train, y_train)

        # 3. Обучение остальных моделей
        trained_models = train_models(X_train, y_train)

        # 4. Добавляем лучший RandomForest в словарь моделей
        trained_models["Best RandomForest"] = best_rf

        # 5. Оценка всех моделей
        print("\n=== Model Evaluation ===")
        for name, model in trained_models.items():
            print(f"\n{name} evaluation:")
            evaluate_model(model, X_test, y_test)

        print("\nЛучшие модели готовы к предсказаниям.")

    elif choice == "2":
        print("Функция предсказания ещё не реализована.")
    elif choice == "3":
        print("Выход из программы.")
        exit()
    else:
        print("Неверная опция, попробуйте снова.")
        main_menu()


if __name__ == "__main__":
    main_menu()
