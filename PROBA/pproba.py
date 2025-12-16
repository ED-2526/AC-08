
# -*- coding: utf-8 -*-
import os
import joblib
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
        "kfold_splits": 5,  # ja no s'utilitza, però el deixo per traçabilitat
        "random_state": 42
    }

    # Diccionario con nombres y rutas de ficheros
    datasets = {
        "PostCrisis": "train_postcrisis.csv",
        "Rang 0-5k": "dataset_rangs/vols_rang_0-5000.csv",
        "Rang 5k-7.5k": "dataset_rangs/vols_rang_5000-7500.csv",
        "Rang 7.5k-10k": "dataset_rangs/vols_rang_7500-10000.csv",
        "Rang 10k-12.5k": "dataset_rangs/vols_rang_10000-12500.csv",
        "Rang 12.5k-15k": "dataset_rangs/vols_rang_12500-15000.csv",
        "Rang gte-15k": "dataset_rangs/vols_rang_gte_15000.csv"
    }

    wandb.init(
        project="Probes",
        name="Proba",
        config=xgb_config
    )

    # carpeta per guardar models
    os.makedirs("models", exist_ok=True)

    # ---------------------------------
    # ENTRENAMENT: un model per dataset
    # ---------------------------------
    for nombre, fichero in datasets.items():
        print(f"Nombre: {nombre} -> Fichero: {fichero}")

        # Carrega i neteja de dades
        df = pd.read_csv(fichero)
        df.dropna(inplace=True)

        # Processat
        df_clean = process_data(df)

        # X i y
        X = df_clean.drop(['Price'], axis=1)
        y = df_clean['Price']

        # Columnes categòriques i numèriques
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X.columns.difference(categorical_cols).tolist()

        # ENTRENAMENT ÚNIC (sense test)
        pipe = train_xgboost(X, y, categorical_cols, numeric_cols)

        # Guardar el model entrenat
        model_path = os.path.join("models", f"{nombre}.joblib")
        joblib.dump(pipe, model_path)
        print(f"Model guardat a: {model_path}")



    # -------------------------------
    # TEST (baseline + routatge)
    # -------------------------------
    test_file = "test_postcrisis.csv"
    if not os.path.exists(test_file):
        print(f"Fitxer de test {test_file} no trobat; saltem la fase d'enrutament")
        wandb.finish()
        return

    df_test = pd.read_csv(test_file)
    df_test.dropna(inplace=True)

    df_test_clean = process_data(df_test)
    # Asegurar que 'Price' es numérico y positivo
    df_test_clean['Price'] = pd.to_numeric(df_test_clean['Price'], errors='coerce')
    df_test_clean = df_test_clean[df_test_clean['Price'].notnull() & (df_test_clean['Price'] > 0)]

    if df_test_clean.empty:
        print(f"No hay datos válidos en {test_file} tras limpieza.")
        wandb.finish()
        return

    X_test = df_test_clean.drop(['Price'], axis=1)
    y_test = df_test_clean['Price'].values

    # Carregar model general
    general_model_path = os.path.join('models', 'PostCrisis.joblib')
    if not os.path.exists(general_model_path):
        print(f"Model general no trobat a {general_model_path}; saltem enrutatge")
        wandb.finish()
        return

    general_pipe = joblib.load(general_model_path)
    print('Model general carregat:', general_model_path)

    # ------------------------
    # Baseline (general, global)
    # ------------------------
    preds_general = general_pipe.predict(X_test)
    preds_general = pd.to_numeric(preds_general, errors='coerce').astype(float)
    mask_general = np.isfinite(preds_general) & (y_test > 0)

    if mask_general.sum() == 0:
        print('No hi ha prediccions vàlides del general per calcular MAE/MAPE.')
    else:
        mae_general = mean_absolute_error(y_test[mask_general], preds_general[mask_general])
        mape_general = np.mean(np.abs((y_test[mask_general] - preds_general[mask_general]) / y_test[mask_general])) * 100
        print(f"Baseline (general) -> MAE: {mae_general:.2f}, MAPE: {mape_general:.2f}%")
        try:
                wandb.log({
                    'mae_general': float(mae_general),
                    'mape_general': float(mape_general)
                })
                print('Estadístiques generals enviades a wandb: mae_general, mape_general')
        except Exception as e:
                print(f"No s'ha pogut enviar a wandb: {e}")

    # --------------------------------------------
    # Routatge fila a fila (global MAE/MAPE)
    # --------------------------------------------
    # Definició de bins i mapping d'etiquetes -> nom de model
    BINS = [
        (0,     5000,  "0-5000"),
        (5000,  7500,  "5000-7500"),
        (7500,  10000, "7500-10000"),
        (10000, 12500, "10000-12500"),
        (12500, 15000, "12500-15000"),
        (15000, None,  "gte_15000"),
    ]
    LABEL_TO_MODEL = {
        "0-5000": "Rang 0-5k",
        "5000-7500": "Rang 5k-7.5k",
        "7500-10000": "Rang 7.5k-10k",
        "10000-12500": "Rang 10k-12.5k",
        "12500-15000": "Rang 12.5k-15k",
        "gte_15000": "Rang gte-15k",
    }

    def assign_label_from_value(val: float):
        """Retorna l’etiqueta del rang segons 'val'; None si NaN/negatiu o fora de rang."""
        if val is None or not np.isfinite(val) or val < 0:
            return None
        for low, high, label in BINS:
            if high is None:
                if val >= low:
                    return label
            else:
                if (val >= low) and (val < high):
                    return label
        return None

    # Acumuladors de traçabilitat
    row_ids = []
    y_true_all = []
    general_pred_all = []
    routed_pred_all = []
    labels_all = []
    loaded_range_models: dict[str, Pipeline] = {}

    # Predicció per fila + routatge
    for i in range(len(X_test)):
        # Mostra com DataFrame d’1 fila (manté columnes per al ColumnTransformer)
        x_row = X_test.iloc[[i]]
        y_row = y_test[i]

        # Predicció amb el general per decidir rang
        gen_pred = general_pipe.predict(x_row)
        gen_pred = float(pd.to_numeric(gen_pred, errors='coerce'))

        label = assign_label_from_value(gen_pred)

        # Si la predicció està a ≤500 del límit del rang, enrutar també al bin adjacent
        label_list = [b[2] for b in BINS]
        candidate_labels = []
        if label is not None and label in label_list:
            idx = label_list.index(label)
            candidate_labels.append(label)
            low, high, _ = BINS[idx]
            # comprovar distància al límit inferior
            if gen_pred - low <= 500 and idx > 0:
                candidate_labels.append(label_list[idx - 1])
            # comprovar distància al límit superior
            if (high is not None) and (high - gen_pred <= 500) and (idx < len(label_list) - 1):
                candidate_labels.append(label_list[idx + 1])
        else:
            # si no hi ha etiqueta, fallback al general
            candidate_labels = []

        # Eliminar duplicats preservant ordre
        seen = set()
        candidate_labels = [x for x in candidate_labels if not (x in seen or seen.add(x))]

        # Prediccions des dels models candidates; si no hi ha prediccions vàlides, fallback al general
        preds_candidates = []
        for cand in candidate_labels:
            if cand not in LABEL_TO_MODEL:
                continue
            model_name = LABEL_TO_MODEL[cand]
            model_path = os.path.join("models", f"{model_name}.joblib")
            # carregar model (cache)
            if cand in loaded_range_models:
                pipe_range = loaded_range_models[cand]
            else:
                if os.path.exists(model_path):
                    try:
                        pipe_range = joblib.load(model_path)
                        loaded_range_models[cand] = pipe_range
                    except Exception as e:
                        print(f"[WARN] No s’ha pogut carregar {model_path}: {e}. Skipping {cand}.")
                        loaded_range_models[cand] = None
                        continue
                else:
                    loaded_range_models[cand] = None
                    continue

            if loaded_range_models[cand] is None:
                continue

            X_row = x_row.copy()
            # injectar columna price_range per a compatibilitat amb alguns pipelines
            X_row['price_range'] = gen_pred
            try:
                p = loaded_range_models[cand].predict(X_row)
                p_arr = pd.to_numeric(p, errors='coerce').astype(float)
                if np.isfinite(p_arr).any():
                    preds_candidates.append(float(np.asarray(p_arr).ravel()[0]))
            except Exception as e:
                print(f"Error predicció amb {model_path} per label {cand} a la mostra {i}: {e}")

        if len(preds_candidates) > 0:
            # usar la mediana de les prediccions candidates
            routed_pred = float(np.nanmedian(np.array(preds_candidates, dtype=float)))
            # per traçabilitat, desar les etiquetes usades (pot ser 1 o 2)
            used_labels = ",".join(candidate_labels)
        else:
            # fallback: usar model general
            routed_pred = float(pd.to_numeric(general_pipe.predict(x_row), errors='coerce'))
            used_labels = label if label is not None else 'general'

        # Guardar traçabilitat
        row_ids.append(X_test.index[i])
        y_true_all.append(y_row)
        general_pred_all.append(gen_pred)
        routed_pred_all.append(routed_pred)
        labels_all.append(label)

    # Arrays per mètriques
    y_true_all = np.array(y_true_all, dtype=float)
    general_pred_all = np.array(general_pred_all, dtype=float)
    routed_pred_all = np.array(routed_pred_all, dtype=float)

    # Màscares de validesa
    mask_routed = np.isfinite(routed_pred_all) & (y_true_all > 0)

    # Mètriques globals del routatge
    if mask_routed.sum() > 0:
        mae_routed_glob = mean_absolute_error(y_true_all[mask_routed], routed_pred_all[mask_routed])
        mape_routed_glob = np.mean(np.abs((y_true_all[mask_routed] - routed_pred_all[mask_routed]) / y_true_all[mask_routed])) * 100
        print(f"[Global] Routatge -> MAE: {mae_routed_glob:.2f}, MAPE: {mape_routed_glob:.2f}%")

        # Log a W&B
        try:
            wandb.log({
                "mae_routed_global_rowwise": float(mae_routed_glob),
                "mape_routed_global_rowwise": float(mape_routed_glob)
            })
            print("Estadístiques routed globals enviades a wandb: mae_routed_global_rowwise, mape_routed_global_rowwise")
        except Exception as e:
            print(f"No s'ha pogut enviar a wandb (rowwise): {e}")
    else:
        print("No hi ha prediccions routades vàlides per calcular MAE/MAPE globals (rowwise).")

    # Guardar CSV de traçabilitat
    df_out = pd.DataFrame({
        "row_id": row_ids,
        "y_true": y_true_all,
        "general_pred": general_pred_all,
        "routed_pred": routed_pred_all,
        "label": labels_all,
    })
    out_path = "prediccions_routatge_rowwise.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Prediccions (general vs. routatge per fila) guardades a: {out_path}")

    # Resum per rang: calcular MAE/MAPE per etiqueta i enviar a wandb
    per_range_metrics = {}
    per_range_counts = {}
    for label in LABEL_TO_MODEL.keys():
        # trobar indices amb aquesta etiqueta
        idxs = [i for i, lab in enumerate(labels_all) if lab == label]
        if len(idxs) == 0:
            continue
        preds_b = routed_pred_all[idxs]
        y_b = y_true_all[idxs]
        valid_mask_b = np.isfinite(preds_b) & (y_b > 0)
        if valid_mask_b.sum() == 0:
            continue
        mae_b = mean_absolute_error(y_b[valid_mask_b], preds_b[valid_mask_b])
        mape_b = np.mean(np.abs((y_b[valid_mask_b] - preds_b[valid_mask_b]) / y_b[valid_mask_b])) * 100
        # clau neta per wandb
        safe_label = label.replace('-', '_').replace(' ', '_')
        per_range_metrics[f"mae_{safe_label}"] = float(mae_b)
        per_range_metrics[f"mape_{safe_label}"] = float(mape_b)
        per_range_counts[safe_label] = int(valid_mask_b.sum())

    if per_range_metrics:
        try:
            wandb.log(per_range_metrics)
            print('Resum per rang enviat a wandb (MAE/MAPE per rang)')
        except Exception as e:
            print(f"No s'ha pogut enviar resum per rang a wandb: {e}")

    # Calcular mitjanes per rang: no ponderada i ponderada per n. Mostres vàlides
    mae_keys = [k for k in per_range_metrics.keys() if k.startswith('mae_')]
    mape_keys = [k for k in per_range_metrics.keys() if k.startswith('mape_')]
    mae_vals = [per_range_metrics[k] for k in mae_keys]
    mape_vals = [per_range_metrics[k] for k in mape_keys]
    # nombre de mostres per label en mateix ordre
    counts = [per_range_counts[k.replace('mae_', '').replace('mape_', '')] for k in mae_keys]

    mean_mae_unweighted = float(np.mean(mae_vals)) if len(mae_vals) > 0 else None
    mean_mape_unweighted = float(np.mean(mape_vals)) if len(mape_vals) > 0 else None

    total_counts = sum(counts) if counts else 0
    if total_counts > 0:
        mean_mae_weighted = float(sum(v * c for v, c in zip(mae_vals, counts)) / total_counts)
        mean_mape_weighted = float(sum(v * c for v, c in zip(mape_vals, counts)) / total_counts)
    else:
        mean_mae_weighted = None
        mean_mape_weighted = None

    print(f"Mitjana MAE (no ponderada) per rangs: {mean_mae_unweighted}")
    print(f"Mitjana MAPE (no ponderada) per rangs: {mean_mape_unweighted}")
    print(f"Mitjana MAE (ponderada) per rangs: {mean_mae_weighted}")
    print(f"Mitjana MAPE (ponderada) per rangs: {mean_mape_weighted}")

    # Enviar mitjanes a wandb
    mean_metrics = {}
    if mean_mae_unweighted is not None:
        mean_metrics['mae_mean_ranges_unweighted'] = mean_mae_unweighted
    if mean_mape_unweighted is not None:
        mean_metrics['mape_mean_ranges_unweighted'] = mean_mape_unweighted
    if mean_mae_weighted is not None:
        mean_metrics['mae_mean_ranges_weighted'] = mean_mae_weighted
    if mean_mape_weighted is not None:
        mean_metrics['mape_mean_ranges_weighted'] = mean_mape_weighted

    if mean_metrics:
        try:
            wandb.log(mean_metrics)
            print('Mitjanes per rangs enviades a wandb')
        except Exception as e:
            print(f"No s'ha pogut enviar mitjanes per rang a wandb: {e}")

    # Comparació amb general (si existeix mape_general)
    try:
        if 'mape_general' in locals() and mean_mape_weighted is not None:
            diff = mape_general - mean_mape_weighted
            diff_pct = (diff / mape_general * 100) if mape_general != 0 else None
            print(f"Diferència MAPE (general - ponderada per rangs): {diff} ({diff_pct}%)")
            try:
                wandb.log({'mape_diff_general_minus_weighted': float(diff)})
                if diff_pct is not None:
                    wandb.log({'mape_diff_pct_general_minus_weighted': float(diff_pct)})
            except Exception:
                pass
    except Exception:
        pass

    # Tancar sessió de W&B
    wandb.finish()


if __name__ == "__main__":
   main()
