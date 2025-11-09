import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Список категориальных признаков
cat_features = ["Make", "Fuel Type", "Transmission", "Owner", "Location", "Seller Type", "Drivetrain"]
# Целевая переменная
target = "Price"

def load_data(path):
    """Загрузить CSV файл с данными"""
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """Очистка и подготовка данных для обучения модели"""
    # Убираем колонки, которые не нужны для модели
    df = df.drop(columns=["Name", "Color", "Model"], errors="ignore")

    # Числовые признаки - очищаем текстовые единицы
    if "Engine" in df.columns:
        df["Engine"] = df["Engine"].str.replace(" cc", "", regex=False).astype(float)

    # Преобразуем Max Power в число (bhp)
    def extract_power(value):
        try:
            return float(value.split(' ')[0])
        except:
            return None

    df['Max Power'] = df['Max Power'].apply(extract_power)

    # Преобразуем Max Torque в число (Nm)
    def extract_torque(value):
        try:
            return float(value.split(' ')[0])
        except:
            return None

    df['Max Torque'] = df['Max Torque'].apply(extract_torque)

    # Проверяем на пропуски
    df = df.dropna()

    # Категориальные признаки → one-hot encoding
    df = pd.get_dummies(df, columns=cat_features, drop_first=True)

    # Разделяем X и y
    X = df.drop(columns=[target])
    y = df[target].values

    return X, y


def split_data(X, y, test_size=0.2):
    """
    Разделение данных на train/test
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)
