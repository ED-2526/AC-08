import pandas as pd
import numpy as np
import wandb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------
# FUNCIONS PROCESAR
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
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4
    }).astype(int)

    df["Route_num_segments"] = df["Route"].apply(lambda x: len(str(x).split(" ? ")))

    festivos_india = ["04/03/2019", "21/03/2019", "06/04/2019", "14/04/2019","17/04/2019", "19/04/2019", "01/05/2019", "18/05/2019", "05/06/2019"]
    df['es_festiu'] = df['Date_of_Journey'].isin(festivos_india).astype(int)

    def get_part_del_dia(h):
        if 0 <= h < 6: return "Madrugada"
        elif 6 <= h < 12: return "Mati"
        elif 12 <= h < 18: return "Tarda"
        else: return "Nit"
    
    df['part_del_dia'] = df['Dep_hour'].apply(get_part_del_dia)

    low_cost_carriers = ["IndiGo", "SpiceJet", "GoAir", "Air Asia", "Trujet"]
    df['es_low_cost'] = df['Airline'].isin(low_cost_carriers).astype(int)

    df['es_cap_de_setmana'] = (df['Journey_weekday'] >= 5).astype(int)

    cols_to_remove = ["Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration"]
    df = df.drop(columns=cols_to_remove)

    return df

# -------------------------------
# MÉTRICAS
# -------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# -------------------------------
# MAIN
# -------------------------------

def main():
    
    
    
    archivo = "Data_Train.csv"
    try:
        df = pd.read_csv(archivo, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: No s'ha trobat l'arxiu {archivo}")
        return
    
    
    df.dropna(inplace=True)
    df_clean = process_data(df)
    
    
    X = df_clean.drop(['Price'], axis=1)
    y = df_clean['Price']
    
    
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.columns.difference(categorical_cols).tolist()
    
    print(f"\nDataset procesat:")
    print(f"- Mostres: {len(X)}")
    print(f"- Features categoriques: {len(categorical_cols)}")
    print(f"- Features numeriquess: {len(numeric_cols)}")
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # -------------------------------
    # CONFIGURAR WANDB
    # -------------------------------
    print("CONFIGURAR WANDB")
    print("Nom del projecte -> enter")
    print("¿Numerar runs per objectiu? -> Aviseu com voleu organitzar wandb")
    
    
    project_name = input("Nom del projecte [Comportament-parametres]: ").strip()
    if not project_name:
        project_name = "Comportament-parametres"
    
    run_name = input("Nom run [xgboost-model]: ").strip()
    if not run_name:
        run_name = "xgboost-model"
    
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model": "XGBRegressor",
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "max_depth": 7,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 5,
            "random_state": 42,
            "tree_method": "hist",
            "test_size": 0.2,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
    )
    
    # -------------------------------
    # ENTRENAR MODEL
    # -------------------------------
    
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )
    
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=5,
        random_state=42,
        tree_method="hist"
    )
    
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb)])
    
    pipe.fit(X_train, y_train)
    
    # -------------------------------
    # EVALUAR Y LOGGEAR
    # -------------------------------
    
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mape(y_train, y_pred_train)
    rmse_train = rmse(y_train, y_pred_train)
    rmsle_train = rmsle(y_train, y_pred_train)
    
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mape(y_test, y_pred_test)
    rmse_test = rmse(y_test, y_pred_test)
    rmsle_test = rmsle(y_test, y_pred_test)
    
    wandb.log({
        "r2_train": r2_train,
        "mae_train": mae_train,
        "mape_train": mape_train,
        "rmse_train": rmse_train,
        "rmsle_train": rmsle_train,
        "r2_test": r2_test,
        "mae_test": mae_test,
        "mape_test": mape_test,
        "rmse_test": rmse_test,
        "rmsle_test": rmsle_test,
        "overfitting_gap_r2": r2_train - r2_test,
        "overfitting_gap_rmse": rmse_train - rmse_test
    })
    
    
    print("Resultats")
    print(f"\nTRAIN:\n  R²: {r2_train:.4f}\n  MAE: {mae_train:.2f}\n  MAPE: {mape_train:.2f}%\n  RMSE: {rmse_train:.2f}\n  RMSLE: {rmsle_train:.4f}")
    print(f"\nTEST:\n  R²: {r2_test:.4f}\n  MAE: {mae_test:.2f}\n  MAPE: {mape_test:.2f}%\n  RMSE: {rmse_test:.2f}\n  RMSLE: {rmsle_test:.4f}")
    print(f"OVERFITTING GAP (RMSE_train - RMSE_test): {rmse_train - rmse_test:.2f}")
    
    # -------------------------------
    # K-FOLD (OPCIONAL)
    # -------------------------------
    
    hacer_kfold = input("\n Vols Fer K-Fold validation(s/n) [n]: ").strip().lower()
    if hacer_kfold == 's':
        try:
            n_folds = int(input("Num folds [5]: ").strip() or "5")
        except:
            n_folds = 5
        
        print(f"\nExecutant K-Fold ({n_folds} folds)...")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        r2_scores = []
        
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train_kf, X_test_kf = X.iloc[train_idx], X.iloc[test_idx]
            y_train_kf, y_test_kf = y.iloc[train_idx], y.iloc[test_idx]
            
            pipe_kf = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.1,
                    max_depth=7,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=5,
                    random_state=42,
                    tree_method="hist"
                ))
            ])
            
            pipe_kf.fit(X_train_kf, y_train_kf)
            y_pred_kf = pipe_kf.predict(X_test_kf)
            r2_kf = r2_score(y_test_kf, y_pred_kf)
            r2_scores.append(r2_kf)
            
            wandb.log({f"kfold/fold_{i+1}_r2": r2_kf})
            print(f"  Fold {i+1}: R² = {r2_kf:.4f}")
        
        avg_r2_kfold = np.mean(r2_scores)
        std_r2_kfold = np.std(r2_scores)
        wandb.log({"kfold/avg_r2": avg_r2_kfold, "kfold/std_r2": std_r2_kfold})
        print(f"\nK-Fold ({n_folds} folds) - R² mitjana: {avg_r2_kfold:.4f} (±{std_r2_kfold:.4f})")
    
    wandb.finish()
    

if __name__ == "__main__":
    main()
