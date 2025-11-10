import pandas as pd

def prompt_float(name, default=None):
    while True:
        try:
            val = input(f"{name}: ")
            if val.strip() == "" and default is not None:
                return default
            return float(val)
        except ValueError:
            print("Введите число.")


def prompt_int(name, default=None):
    while True:
        try:
            val = input(f"{name}: ")
            if val.strip() == "" and default is not None:
                return default
            return int(val)
        except ValueError:
            print("Введите целое число.")


def prompt_choice(name, choices, default=None):
    choices = list(choices)
    print(f"{name} варианты: {', '.join(choices)}")
    while True:
        val = input(f"{name}: ").strip()
        if val == "" and default is not None:
            return default
        if val in choices:
            return val
        print("Неверный выбор, попробуйте снова.")


def make_feature_row(columns_template):
    """
    Формирует корректную one-hot строку под сохранённый шаблон колонок.
    Спрашивает только те признаки, которые реально есть в шаблоне.
    """
    base = {}
    if "Year" in columns_template: base["Year"] = prompt_int("Year", 2015)
    if "Kilometers Driven" in columns_template: base["Kilometers Driven"] = prompt_int("Kilometers Driven", 50000)
    if "Mileage" in columns_template: base["Mileage"] = prompt_float("Mileage (kmpl)", 18.0)
    if "Engine" in columns_template: base["Engine"] = prompt_float("Engine (cc)", 1200.0)
    if "Max Power" in columns_template: base["Max Power"] = prompt_float("Max Power (bhp)", 80.0)
    if "Max Torque" in columns_template: base["Max Torque"] = prompt_float("Max Torque (Nm)", 120.0)
    if "Seats" in columns_template: base["Seats"] = prompt_int("Seats", 5)

    # Сгруппируем one-hot по префиксам 'Feature_Value'
    cats = {}
    for col in columns_template:
        if "_" in col:
            prefix = col.split("_")[0]
            cats.setdefault(prefix, set()).add(col)

    for prefix, cols in cats.items():
        values = [c[len(prefix) + 1:] for c in cols]
        if values:
            choice = prompt_choice(prefix, values, default=values[0])
            base[f"{prefix}_{choice}"] = 1

    # Сформируем строку признаков под точные имена столбцов
    row = {c: 0 for c in columns_template}
    for k, v in base.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row])
