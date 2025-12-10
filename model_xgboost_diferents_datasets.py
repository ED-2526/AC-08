
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, median_absolute_error, max_error
)

from xgboost import XGBRegressor
import wandb

# -------------------------------
# PREPROCESSAT
# -------------------------------
def adjust_duration(duration: str) -> str:
    if len(str(duration).split()) != 2:
        duration = str(duration)
        if "h" in duration:
            duration = duration.strip() + " 0m"
        else:
            duration = "0h " + duration
    return duration

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    fecha_dt = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_day"] = fecha_dt.dt.day
    df["Journey_month"] = fecha_dt.dt.month
    df["Journey_weekday"] = fecha_dt.dt.dayofweek

    df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.hour
    df["Dep_min"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.minute

    # Si "Arrival_Time" té format mixt (p.ex. inclou data), el parseig sense format és més robust
    df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
    df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute

    duration = [adjust_duration(x) for x in df["Duration"]]
    duration_hours = [int(d.split("h")[0]) for d in duration]
    duration_mins = [int(d.split("m")[0].split()[-1]) for d in duration]
    df["Duration_Total_Mins"] = np.array(duration_hours) * 60 + np.array(duration_mins)

    df["Total_Stops"] = df["Total_Stops"].replace({
        "non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4
    }).astype(int)

    # Nombre de segments de la ruta (split pel separador " ? ")
    df["Route_num_segments"] = df["Route"].apply(lambda x: len(str(x).split(" ? ")))

    festius_india = ["04/03/2019", "21/03/2019", "06/04/2019", "14/04/2019",
                     "17/04/2019", "19/04/2019", "01/05/2019", "18/05/2019", "05/06/2019"]
    df['es_festiu'] = df['Date_of_Journey'].isin(festius_india).astype(int)

    def get_part_del_dia(h):
        if 0 <= h < 6: return "Madrugada"
        elif 6 <= h < 12: return "Matí"
        elif 12 <= h < 18: return "Tarda"
        else: return "Nit"

    df['part_del_dia'] = df['Dep_hour'].apply(get_part_del_dia)

    low_cost_carriers = ["IndiGo", "SpiceJet", "GoAir", "Air Asia", "Trujet"]
    df['es_low_cost'] = df['Airline'].isin(low_cost_carriers).astype(int)
    df['es_cap_de_setmana'] = (df['Journey_weekday'] >= 5).astype(int)

    df = df.drop(columns=["Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration"])
    return df

# -------------------------------
# ENTRENAMENT: NOMÉS XGBOOST
# -------------------------------
def train_xgboost(X_train, y_train, categorical_cols, numeric_cols):
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.55,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        tree_method="hist",  # ràpid i estable en dades tabulars
        n_jobs=-1
    )

    pipe = Pipeline([('preprocessor', preprocessor), ('model', xgb)])
    pipe.fit(X_train, y_train)
    return pipe

# -------------------------------
# CROSS-VALIDATION I MÈTRIQUES
# -------------------------------
def cross_val_model(train_func, X, y, categorical_cols, numeric_cols, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_dict = {
        "MAE": [], "MAPE": [], "MSE": [], "RMSE": [], "MedAE": [], "MaxError": []
    }
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = train_func(X_train, y_train, categorical_cols, numeric_cols)
        y_pred = pipe.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        metrics_dict["MAE"].append(mean_absolute_error(y_test, y_pred))
        # MAPE (assumim preus > 0). Si hi hagués 0, caldria un MAPE ajustat.
        metrics_dict["MAPE"].append(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
        metrics_dict["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_dict["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_dict["MedAE"].append(median_absolute_error(y_test, y_pred))
        metrics_dict["MaxError"].append(max_error(y_test, y_pred))

    summary = {metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))} 
               for metric, vals in metrics_dict.items()}

    return summary, np.array(y_true_all), np.array(y_pred_all)

# -------------------------------
# MAIN
# -------------------------------
def main():
    # Hiperparàmetres d'XGBoost al config de wandb
    xgb_config = {
        "model": "XGBRegressor",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 9,
        "subsample": 0.8,
        "colsample_bytree": 0.55,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "tree_method": "hist",
        "kfold_splits": 5,
        "random_state": 42
    }

    
    wandb.init(
        project="Analisis_dataset_qunitils_rangs",
        name="3.5 Dataset_Quintils_Part_5",
        config=xgb_config
    )

    # Carrega i neteja de dades
    df = pd.read_csv("dataset_quintils/vols_part_5.csv")
    df.dropna(inplace=True)
    df_clean = process_data(df)

    # X i y
    X = df_clean.drop(['Price'], axis=1)
    y = df_clean['Price']

    # Columnes categòriques i numèriques
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.columns.difference(categorical_cols).tolist()

    # Cross-validation només amb XGBoost
    summary, y_true, y_pred = cross_val_model(
        train_xgboost, X, y, categorical_cols, numeric_cols, n_splits=wandb.config["kfold_splits"]
    )

    # Log de mètriques (mitjana i desviació)
    for metric, stats in summary.items():
        wandb.log({
            f"XGBoost/{metric}_mean": stats["mean"],
            f"XGBoost/{metric}_std": stats["std"]
        })

    # Gràfic y_true vs y_pred
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', label="Identitat")
    ax.set_title("XGBoost: y_true vs y_pred")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.legend()
    wandb.log({"XGBoost/y_true_vs_y_pred": wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()

if __name__ == "__main__":
    main()
