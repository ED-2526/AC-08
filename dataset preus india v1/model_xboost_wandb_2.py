import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------
# 1. FUNCIONS DE PROCESAMENT
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
    # Dates
    fecha_dt = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_day"] = fecha_dt.dt.day
    df["Journey_month"] = fecha_dt.dt.month
    df["Journey_weekday"] = fecha_dt.dt.dayofweek

    # Hores
    df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.hour
    df["Dep_min"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.minute
    df["Arrival_hour"] = pd.to_datetime(df.Arrival_Time).dt.hour
    df["Arrival_min"] = pd.to_datetime(df.Arrival_Time).dt.minute

    # Duració
    duration = [adjust_duration(x) for x in df["Duration"]]
    duration_hours = [int(d.split("h")[0]) for d in duration]
    duration_mins = [int(d.split("m")[0].split()[-1]) for d in duration]
    df["Duration_Total_Mins"] = np.array(duration_hours) * 60 + np.array(duration_mins)

    # Escales
    df["Total_Stops"] = df["Total_Stops"].replace({
        "non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4
    }).astype(int)

    # Variables Extres
    festivos_india = ["04/03/2019", "21/03/2019", "06/04/2019", "14/04/2019","17/04/2019", "19/04/2019", "01/05/2019", "18/05/2019", "05/06/2019"]
    df['es_festiu'] = df['Date_of_Journey'].isin(festivos_india).astype(int)

    low_cost_carriers = ["IndiGo", "SpiceJet", "GoAir", "Air Asia", "Trujet"]
    df['es_low_cost'] = df['Airline'].isin(low_cost_carriers).astype(int)

    def get_part_of_day(h):
        if 0 <= h < 6: return "Madrugada"
        elif 6 <= h < 12: return "Mati"
        elif 12 <= h < 18: return "Tarda"
        else: return "Nit"
    df['part_del_dia'] = df['Dep_hour'].apply(get_part_of_day)

    # Neteja
    cols_to_remove = ["Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration"]
    df = df.drop(columns=cols_to_remove)

    return df

# -------------------------------
# 2. FUNCIONS D'ANÀLISI
# -------------------------------

def analizar_error_por_duracion(X_test, y_test, y_pred, run_name):
    """
    Calcula MAE i RMSE per a diferents duracions de vol.
    """
    analysis_df = X_test.copy()
    analysis_df['Absolute_Error'] = np.abs(y_test - y_pred)
    
    # Buckets de duración
    bins = [0, 180, 360, 720, 5000]
    labels = ['Curt (<3h)', 'Mitjà (3-6h)', 'Llarg (6-12h)', 'Molt Llarg (>12h)']
    
    analysis_df['Duration_Category'] = pd.cut(analysis_df['Duration_Total_Mins'], bins=bins, labels=labels)
    
    # Calcular MAE per grupo
    mae_by_group = analysis_df.groupby('Duration_Category')['Absolute_Error'].mean()
    
    # Loggejar a WandB
    data = [[label, val] for label, val in zip(mae_by_group.index, mae_by_group.values)]
    table = wandb.Table(data=data, columns=["Duracio", "MAE (Error Diners)"])
    wandb.log({f"Error_Segons_Duracio": wandb.plot.bar(table, "Duracio", "MAE (Error Diners)", title=f"Error ({run_name})")})

# -------------------------------
# 3. MOTOR D'EXPERIMENTS
# -------------------------------

def ejecutar_experimento(param_name, param_value, X_train, y_train, X_test, y_test, categorical_cols, numeric_cols):
    
    # Configuració base (Valores por defecto)
    config = {
        "n_estimators": 500,
        "learning_rate": 0.1,
        "max_depth": 7
    }
    
    # Sobreescriu el paràmetre que estem probant
    config[param_name] = param_value
    
    # Inicia WandB
    run_name = f"XGB_{param_name}_{param_value}"
    run = wandb.init(
        project="Vols-India-Experiments-MAE",
        name=run_name,
        reinit=True,
        config=config
    )
    
    print(f"\nTEST: {param_name} = {param_value} ...")

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )
    
    xgb = XGBRegressor(
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        random_state=42,
        n_jobs=-1
    )
    
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb)])
    
    # Entrenament
    try:
        pipe.fit(X_train, y_train)
        
        # Prediccions (Train i Test)
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        
        # Mètrica Principal: MAE (Diners)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        # Mètrica Secundaria: R2
        r2_test = r2_score(y_test, y_pred_test)
        
        # CÀLCUL OVERFITTING (Basat en MAE)
        # Gap
        overfitting_gap_mae = mae_test - mae_train 
        
        print(f"MAE Test: {mae_test:.2f} | MAE Train: {mae_train:.2f}")
        print(f"Overfitting Gap: {overfitting_gap_mae:.2f} rupias")

        # Log a WandB
        wandb.log({
            f"{param_name}": param_value,
            "MAE_Test": mae_test,
            "MAE_Train": mae_train,
            "Overfitting_Gap_MAE": overfitting_gap_mae,
            "R2_Test": r2_test
        })
        
        # Anàlisi extra només si el model no ha explotat (MAE razonable)
        if mae_test < 20000:
            analizar_error_por_duracion(X_test, y_test, y_pred_test, run_name)
            
    except Exception as e:
        print(f"El model ha fallat amb aquests paràmetres. Error: {e}")

    run.finish()

# -------------------------------
# 4. MAIN
# -------------------------------

def main():
    archivo = "Data_Train.csv"
    try:
        df = pd.read_csv(archivo, encoding='utf-8')
    except:
        df = pd.read_csv(archivo, encoding='latin1') 
    
    df.dropna(inplace=True)
    df_clean = process_data(df)
    
    X = df_clean.drop(['Price'], axis=1)
    y = df_clean['Price']
    
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.columns.difference(categorical_cols).tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==============================================================================
    # EXPERIMENTS
    # ==============================================================================
    
    # 1. LEARNING RATE (De 0.01 a 1000)
    # ----------------------------------------------------
    print("\n>>> EXPERIMENT 1: LEARNING RATE SWEEP")
    learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0, 500.0, 1000.0]
    
    for lr in learning_rates:
        ejecutar_experimento("learning_rate", lr, X_train, y_train, X_test, y_test, categorical_cols, numeric_cols)

    # 2. N_ESTIMATORS (De 10 a 2000)
    # ----------------------------------------------------
    print("\n>>> EXPERIMENT 2: N_ESTIMATORS SWEEP")
    # Escala logarítmica similar.
    n_estimators_list = [10, 50, 100, 200, 500, 1000, 2000]
    
    for n_est in n_estimators_list:
        ejecutar_experimento("n_estimators", n_est, X_train, y_train, X_test, y_test, categorical_cols, numeric_cols)

if __name__ == "__main__":
    main()