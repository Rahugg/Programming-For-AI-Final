import os
import numpy as np
import pandas as pd


def residuals(y_true, y_pred):
    return y_true - y_pred


def error_summary(y_true, y_pred, top_n=10):
    res = residuals(y_true, y_pred)
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": res,
        "abs_err": np.abs(res)
    })
    df_sorted = df.sort_values("abs_err", ascending=False)
    tops = df_sorted.head(top_n)
    return df, tops


def save_error_report(df_full, df_top, out_dir="reports", base_name="error_report"):
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, f"{base_name}_full.csv")
    top_path = os.path.join(out_dir, f"{base_name}_top.csv")
    df_full.to_csv(full_path, index=False)
    df_top.to_csv(top_path, index=False)
    return full_path, top_path
