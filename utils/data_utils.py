import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Возможные имена целевой переменной в разных версиях Cardekho
target_candidates = ["Price", "selling_price", "Selling Price", "price"]


def detect_target(df: pd.DataFrame) -> str:
    for c in target_candidates:
        if c in df.columns:
            return c
    raise ValueError("Не удалось найти колонку с целевой переменной (цена).")


def load_data(path: str) -> pd.DataFrame:
    """Загрузить CSV файл с данными."""
    return pd.read_csv(path)


def _clean_numeric(series: pd.Series, suffix: str = None) -> pd.Series:
    """
    Преобразовать текстовые числа вида '1,197 cc' / '82.5 bhp' / '23.4 kmpl' в float.
    Усечёт любые нецифровые хвосты; безопасно вернёт NaN при невозможности парсинга.
    """
    if series.dtype != object:
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str)
    if suffix is not None:
        s = s.str.replace(f" {suffix}", "", regex=False)

    # извлекаем первое число (целое/вещественное)
    s = s.str.extract(r"([0-9]*\.?[0-9]+)")[0]
    return pd.to_numeric(s, errors="coerce")


def preprocess_data(df: pd.DataFrame):
    """
    Устойчивый препроцессинг:
      1) нормализация ключевых названий (Year, Kilometers Driven и т.д.);
      2) мягкое удаление высококардинальных/нерелевантных полей;
      3) парсинг числовых полей (Engine/Power/Torque/Mileage/...);
      4) универсальное one-hot кодирование ВСЕХ категориальных признаков;
      5) санитарная обработка (±inf/NaN, zero-variance, клиппинг выбросов 0.5–99.5%);
      6) возврат X, y, columns_template.
    """
    df = df.copy()

    # ---- 1) Нормализуем названия распространённых столбцов
    rename_map = {
        "Year": "Year", "year": "Year",
        "KM Driven": "Kilometers Driven", "km_driven": "Kilometers Driven",
        "Mileage": "Mileage", "mileage": "Mileage",
        "Engine": "Engine", "engine": "Engine",
        "Max Power": "Max Power", "max_power": "Max Power",
        "Torque": "Max Torque", "torque": "Max Torque",
        "Seats": "Seats", "seats": "Seats",
        "Fuel": "Fuel Type", "fuel": "Fuel Type",
        "Transmission": "Transmission", "transmission": "Transmission",
        "Owner": "Owner", "owner": "Owner",
        "Seller Type": "Seller Type", "seller_type": "Seller Type",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Специальный случай: «ломаные» заголовки пробега (например, 'Kilometer ...in')
    if "Kilometers Driven" not in df.columns:
        for c in df.columns:
            name = str(c).lower().replace("…", "...")
            if "kilometer" in name and "driv" in name:
                df = df.rename(columns={c: "Kilometers Driven"})
                break

    # ---- 2) Целевая переменная
    target = detect_target(df)
    if target != "Price":
        df = df.rename(columns={target: "Price"})

    # ---- 3) Удаляем нерелевантные/высококардинальные поля (базовый набор)
    for c in ["Name", "Color", "Model", "Location", "Drivetrain"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # ---- 4) Парсинг чисел
    if "Engine" in df.columns:
        df["Engine"] = _clean_numeric(df["Engine"], "cc")
    if "Max Power" in df.columns:
        df["Max Power"] = _clean_numeric(df["Max Power"])
    if "Max Torque" in df.columns:
        df["Max Torque"] = _clean_numeric(df["Max Torque"])
    if "Mileage" in df.columns:
        df["Mileage"] = _clean_numeric(df["Mileage"])
    if "Seats" in df.columns:
        df["Seats"] = pd.to_numeric(df["Seats"], errors="coerce")
    if "Kilometers Driven" in df.columns:
        df["Kilometers Driven"] = pd.to_numeric(df["Kilometers Driven"], errors="coerce")

    # Уберём строки с NaN в Price на ранней стадии
    df = df[df["Price"].notna()]

    # ---- 5) Универсальное one-hot кодирование ВСЕХ категориальных фич
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if "Price" in cat_cols:
        cat_cols.remove("Price")
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # ---- 6) Санитарная обработка числовой матрицы (без chained assignment)
    # Приведём к float64
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    df[num_cols] = df[num_cols].astype("float64")

    # Заменим ±inf → NaN (без inplace)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Заполним NaN в числовых столбцах медианой (если столбец весь NaN — удалим)
    for c in num_cols:
        if df[c].isna().any():
            med = df[c].median()
            if np.isnan(med):
                df = df.drop(columns=[c])
            else:
                # ВАЖНО: без inplace на срезе
                df[c] = df[c].fillna(med)

    # Удалим столбцы с нулевой дисперсией (константы)
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    zero_var = df[num_cols].std(ddof=0)
    zero_var_cols = zero_var[zero_var == 0].index.tolist()
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)

    # Мягкий клиппинг выбросов: 0.5%..99.5% по каждому числовому столбцу
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if num_cols:
        low_q = df[num_cols].quantile(0.005)
        hi_q = df[num_cols].quantile(0.995)
        df[num_cols] = df[num_cols].clip(lower=low_q, upper=hi_q, axis=1)

    # Финальная проверка на конечность
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if not np.isfinite(df[num_cols].to_numpy(dtype="float64")).all():
        raise ValueError("Обнаружены не-конечные значения после препроцессинга.")

    # ---- 7) Возвращаем X, y, шаблон колонок
    X = df.drop(columns=["Price"])
    y = df["Price"].to_numpy(dtype="float64")
    columns_template = list(X.columns)
    return X, y, columns_template


def split_data(X, y, test_size=0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
