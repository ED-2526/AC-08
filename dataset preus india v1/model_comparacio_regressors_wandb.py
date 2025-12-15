import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, max_error
import wandb
import matplotlib.pyplot as plt

# -------------------------------
# PREPROCESSAMENT
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
    df["Arrival_hour"] = pd.to_datetime(df.Arrival_Time).dt.hour
    df["Arrival_min"] = pd.to_datetime(df.Arrival_Time).dt.minute
    duration = [adjust_duration(x) for x in df["Duration"]]
    duration_hours = [int(d.split("h")[0]) for d in duration]
    duration_mins = [int(d.split("m")[0].split()[-1]) for d in duration]
    df["Duration_Total_Mins"] = np.array(duration_hours) * 60 + np.array(duration_mins)
    df["Total_Stops"] = df["Total_Stops"].replace({
        "non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4
    }).astype(int)
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
# ENTRENAMENT
# -------------------------------
def train_random_forest(X_train, y_train, categorical_cols, numeric_cols):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ])
    rf_model = RandomForestRegressor(
        n_estimators=250, max_depth=25, min_samples_split=5,
        min_samples_leaf=1, max_features=0.7, max_samples=0.9,
        bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
    )
    pipe = Pipeline([("pre", preprocessor), ("model", rf_model)])
    pipe.fit(X_train, y_train)
    return pipe

def train_xgboost(X_train, y_train, categorical_cols, numeric_cols):
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])
    xgb = XGBRegressor(
        n_estimators=2000, learning_rate=0.04, max_depth=7,
        subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5,
        reg_lambda=5, random_state=42, tree_method="hist"
    )
    pipe = Pipeline([('preprocessor', preprocessor), ('model', xgb)])
    pipe.fit(X_train, y_train)
    return pipe

def train_svr(X_train, y_train, categorical_cols, numeric_cols):
    y_train_log = np.log1p(y_train)
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    svr_model = SVR(kernel='rbf', C=10, epsilon=0.05, gamma='scale')
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=30, random_state=42)),
        ('scaler', StandardScaler()),
        ('model', svr_model)
    ])
    pipe.fit(X_train, y_train_log)
    return pipe

# -------------------------------
# CROSS-VALIDATION I MÈTRIQUES
# -------------------------------
def cross_val_model(train_func, X, y, categorical_cols, numeric_cols, model_name, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_dict = {
        "MAE": [], "MAPE": [], "MSE": [], "RMSE": [], "MedAE": [], "MaxError": []
    }
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = train_func(X_train, y_train, categorical_cols, numeric_cols)

        if model_name=="SVR":
            y_pred_log = pipe.predict(X_test)
            y_pred = np.expm1(y_pred_log)
        else:
            y_pred = pipe.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        metrics_dict["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_dict["MAPE"].append(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
        metrics_dict["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_dict["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_dict["MedAE"].append(median_absolute_error(y_test, y_pred))
        metrics_dict["MaxError"].append(max_error(y_test, y_pred))

    summary = {metric: {"mean": np.mean(vals), "std": np.std(vals)} for metric, vals in metrics_dict.items()}

    return summary, np.array(y_true_all), np.array(y_pred_all)

# -------------------------------
# MAIN
# -------------------------------
def main():
    wandb.init(project="comparacio_regressors", config={"kfold_splits": 5})

    df = pd.read_csv("Data_Train.csv")
    df.dropna(inplace=True)
    df_clean = process_data(df)
    X = df_clean.drop(['Price'], axis=1)
    y = df_clean['Price']
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.columns.difference(categorical_cols).tolist()

    models = {
        "RandomForest": train_random_forest,
        "XGBoost": train_xgboost,
        "SVR": train_svr
    }

    all_metrics = {}

    for name, train_func in models.items():
        summary, y_true, y_pred = cross_val_model(train_func, X, y, categorical_cols, numeric_cols, name)
        all_metrics[name] = summary

        # Gràfic y_true vs y_pred
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        ax.set_title(f"{name}: y_true vs y_pred")
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        wandb.log({f"{name}/y_true_vs_y_pred": wandb.Image(fig)})
        plt.close(fig)

    # Comparativa de mètriques
    metrics_to_plot = ["MAE", "MAPE", "MSE", "RMSE", "MedAE", "MaxError"]
    for metric in metrics_to_plot:
        means = [all_metrics[m][metric]["mean"] for m in models]
        stds = [all_metrics[m][metric]["std"] for m in models]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(models.keys(), means, yerr=stds, capsize=5, color=['skyblue','salmon','lightgreen'])
        ax.set_title(f"Comparativa {metric} (mitjana ± desviació estàndard)")
        ax.set_ylabel(metric)
        wandb.log({f"Comparativa/{metric}": wandb.Image(fig)})
        plt.close(fig)

    wandb.finish()

if __name__ == "__main__":
    main()
