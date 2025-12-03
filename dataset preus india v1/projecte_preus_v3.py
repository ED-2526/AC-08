
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# FUNCIONS DE PROCESSAMENT
# -------------------------------

def adjust_duration(duration: str) -> str:
    ## Aquesta funció assegura que la duració tingui el format 'Xh Ym'.
    ## Si només té hores o minuts, afegeix la part que falta per evitar errors.
    if len(str(duration).split()) != 2:
        duration = str(duration)
        if "h" in duration:
            duration = duration.strip() + " 0m"
        else:
            duration = "0h " + duration
    return duration

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    ## Crea noves variables numèriques a partir de les columnes originals
    ## i deixa les categòriques per al ColumnTransformer.

    ## Converteix la data del viatge en dia, mes i dia de la setmana.
    fecha_dt = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_day"] = fecha_dt.dt.day
    df["Journey_month"] = fecha_dt.dt.month
    df["Journey_weekday"] = fecha_dt.dt.dayofweek

    ## Dep_Time sempre té format %H:%M, així que podem parsejar-lo directament.
    df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.hour
    df["Dep_min"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.minute

    ## Arrival_Time pot tenir formats mixtos (%H:%M o %H:%M + dia),
    ## per això no li específiquem ningun format i pandas ho interpreta automàticament.
    df["Arrival_hour"] = pd.to_datetime(df.Arrival_Time).dt.hour
    df["Arrival_min"] = pd.to_datetime(df.Arrival_Time).dt.minute

    ## Converteix la duració (ex. '2h 50m') en minuts totals.
    duration = [adjust_duration(x) for x in df["Duration"]]
    duration_hours = [int(d.split("h")[0]) for d in duration]
    duration_mins = [int(d.split("m")[0].split()[-1]) for d in duration]
    df["Duration_Total_Mins"] = np.array(duration_hours) * 60 + np.array(duration_mins)

    ## Converteix el text (ex. 'non-stop', '2 stops') en enters.
    df["Total_Stops"] = df["Total_Stops"].replace({
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4
    }).astype(int)

    ## Route → nombre de segments (comptant separador " ? "), en el csv separa els diferents segments amb " ? "
    df["Route_num_segments"] = df["Route"].apply(lambda x: len(str(x).split(" ? ")))

    ## Nova variable: es_festiu
    ## Llista manual de dies fesius a la india (de Març a Juny)
    festius_india = ["04/03/2019", "21/03/2019", "06/04/2019", "14/04/2019","17/04/2019", "19/04/2019", "01/05/2019", "18/05/2019", "05/06/2019"]
    df['es_festiu'] = df['Date_of_Journey'].isin(festius_india).astype(int)

    ## Nova variable: part_del_dia
    ## Segons la hora del dia classifica segons Madrugada, Matí, Tarda o Nit
    def get_part_del_dia(h):
        if 0 <= h < 6: return "Madrugada"     # 00:00 - 05:59
        elif 6 <= h < 12: return "Mati"    # 06:00 - 11:59
        elif 12 <= h < 18: return "Tarda" # 12:00 - 17:59
        else: return "Nit"                  # 18:00 - 23:59
    df['part_del_dia'] = df['Dep_hour'].apply(get_part_del_dia)
    # Al ser variable categòrica el OneHotEncoder s'aplicarà automàticament

    ## Nova variable: es_low_cost
    ## Definim en una llista manual les aerolinies que son considerades low cost a la India
    low_cost_carriers = ["IndiGo", "SpiceJet", "GoAir", "Air Asia", "Trujet"]
    df['es_low_cost'] = df['Airline'].isin(low_cost_carriers).astype(int)

    ## Nova variable: es_cap_de_setmana
    ## Si el dia és Dissabte (5) o Diumenge (6) -> 1, sino -> 0
    df['es_cap_de_setmana'] = (df['Journey_weekday'] >= 5).astype(int)

    ## Eliminem les variables derivades redundants que són text.
    cols_to_remove = ["Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration"]
    df = df.drop(columns=cols_to_remove)

    return df


# -------------------------------
# FUNCIONS DE MODELATGE
# -------------------------------
def train_random_forest(X_train, y_train, categorical_cols, numeric_cols):
    ## Crea un ColumnTransformer per aplicar:
    ## - StandardScaler a les variables numèriques
    ## - OneHotEncoder a les categòriques
    """    
    ## Per què fem servir ColumnTransformer?
    ## Ens permet aplicar diferents transformacions a diferents tipus de columnes en un sol pas:
    ## - StandardScaler per a variables numèriques (normalitza escales)
    ## - OneHotEncoder per a variables categòriques (converteix categories en columnes binàries)
    ## Això evita fer encoding i escalat manualment i assegura un pre-processament consistent.
    """

    """
    ## Per què fem servir OneHotEncoder?
    ## Els models com RandomForest només accepten valors numèrics.
    ## Si deixem les categories com a text, el model no pot treballar amb elles.
    ## LabelEncoder assignaria números (0,1,2...) però això imposa un ordre fictici.
    ## OneHotEncoder crea columnes binàries per cada categoria (0/1),
    ## evitant ordres falsos i permetent al model tractar cada categoria de forma independent.
    ## Exemple:
        ## Airline: Jet Airways, IndiGo → Airline_Jet Airways | Airline_IndiGo
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    # Corregir overfitting:
    # - max_depth: Posa un sostre a l'alçada de l'arbre (abans era infinita).
    # - min_samples_leaf: obliga que cada decisió es base en almenys x vols.
    # - min_samples_split: No divideix una branca si te menys de x dades.
    rf_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1  # Utilitza tots els nuclis del PC
    )

    """
    ## Per què fem servir Pipeline?
    ## Combina el pre-processament (ColumnTransformer) i el model (RandomForest) en un únic objecte.
    ## Avantatges:
    ## - Flux net i ordenat: entrenament i predicció amb una sola crida (fit/predict)
    ## - Evita errors en aplicació de transformacions a train/test
    ## - Facilita guardar i reutilitzar el model complet amb joblib
    """
    ## Pipeline amb el pre-processament i el model RandomForest
    pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", rf_model)
    ])

    ## Entrena el pipeline amb les dades d'entrenament
    pipe.fit(X_train, y_train)
    return pipe

def regresion_avaluation(y_test, y_pred):
    ## Calcula mètriques de rendiment:
    ## - R²: proporció de variabilitat explicada pel model
    ## - MAE: error absolut mitjà
    ## - MAPE: error percentual mitjà
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print("-" * 50)
    print(f"Resultats finals del model:")
    print(f"R2 Score (Nota):   {r2:.4f}")
    print(f"MAE (Error Mitjà): {mae:.2f}")
    print(f"MAPE (Error %):    {mape:.2f}%")
    print("-" * 50)

    # Gràfiques d'evaluació
    plt.figure(figsize=(14, 6))

    # Realitat vs Predicció (Scatter)
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='blue', label='Prediccions')
    
    # Línea de perfecció (Diagonal)
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'r--', lw=2, label='Perfecció ideal')
    
    plt.title(f'Precissió: Realitat vs Predicció (R2: {r2:.2f})')
    plt.xlabel('Preu Real')
    plt.ylabel('Preu Predit')
    plt.legend()

    # Histograma de Residus (Errors)
    # Si la curva és una campana centrada en 0, el model és honest.
    plt.subplot(1, 2, 2)
    residuos = y_test - y_pred
    sns.histplot(residuos, kde=True, color='purple')
    plt.axvline(x=0, color='red', linestyle='--', lw=2)
    plt.title('Distribució d Errors (Residus)')
    plt.xlabel('Diferència de Preu (Error)')
    
    plt.tight_layout()
    plt.show()

def view_var_importance(pipe, categorical_cols, numeric_cols):
    ## Extreu els noms de les variables després del OneHotEncoder
    ## per interpretar la importància de les features.
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = numeric_cols + list(cat_feature_names)

    ## Calcula la importància de cada feature segons RandomForest
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
    print("\nTop 15 variables més importants:")
    print(fi.head(15))

# -------------------------------
# MAIN
# -------------------------------
def main():
    ## Evita avisos de futur canvi en pandas
    #pd.set_option('future.no_silent_downcasting', True)

    ## Carrega el dataset des de CSV
    archivo = "Data_Train.csv"
    df = pd.read_csv(archivo, encoding='utf-8')
    df.dropna(inplace=True)

    ## Processa les dades (afegeix noves features i elimina columnes redundants)
    df_clean = process_data(df)

    """
    ## Selecció de variables
    ## Elimina les variables que van quedar últimes en el rànking de variables més importants
    vars_a_eliminar = ['Dep_min', 'Arrival_min','Route_num_segments','part_del_dia','es_festiu']
    df_clean = df_clean.drop(columns=vars_a_eliminar, errors='ignore')
    """

    ## Separa X (features) i y (target)
    X = df_clean.drop(['Price'], axis=1)
    y = df_clean['Price']

    ## Detecta columnes categòriques i numèriques
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.columns.difference(categorical_cols).tolist()

    ## Divideix en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ## Entrena el pipeline amb RandomForest
    pipe = train_random_forest(X_train, y_train, categorical_cols, numeric_cols)

    ## Prediccions sobre el conjunt de test
    y_pred = pipe.predict(X_test)

    ## Avalua el model amb el conjunt de Test
    regresion_avaluation(y_test, y_pred)

    ## Comprova Overfitting
    print("\nAnàlisi Overfitting")
    score_train = pipe.score(X_train, y_train)
    score_test = pipe.score(X_test, y_test)
    print(f"R2 Train: {score_train:.4f} | R2 Test: {score_test:.4f}")
    
    # Bretxa entre train y test
    gap = score_train - score_test
    
    if gap > 0.10:
        print(f"Overfitting alt (Diferencia: {gap:.2%}). El model memoriza.")
    elif gap < 0.05:
        print(f"Bona generalització (Diferencia: {gap:.2%}).")
    else:
        print(f"Overfitting moderat/acceptat (Diferencia: {gap:.2%}).")


    ## Mostra les variables més importants segons el model
    view_var_importance(pipe, categorical_cols, numeric_cols)

if __name__ == "__main__":
    main()
