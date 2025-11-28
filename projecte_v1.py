## projecte_v1.py - Model de classificació per predir cancel·lacions de vols

## Instal·lar dependències (si cal): kaggle pandas scikit-learn xgboost matplotlib fastparquet
## Arxius de dades: https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018?resource=download
## Fitxers CSV descarregats a la carpeta 'data/'

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

## Ruta del fitxer Parquet (per càrrega ràpida)
PARQUET_PATH = "data/vols_2009_2018.parquet"

## Carregar dades des del Parquet si existeix
if os.path.exists(PARQUET_PATH):
    print("Carregant dataset des del fitxer Parquet...")
    df = pd.read_parquet(PARQUET_PATH)
## Cas contrari, si no existeix el fitxer Parquet, crear-lo a partir dels CSVs individuals
else:
    ## Unificem els CSVs
    import glob
    files = glob.glob("data/*.csv")
    dfs = []

    for f in files:
        print("Carregant dataset:", f)
        df_year = pd.read_csv(f, low_memory=False)

        ## Eliminar columnes sobrants i capçaleres duplicades entre els anys
        df_year = df_year.loc[:, ~df_year.columns.str.contains('^Unnamed')]
        df_year = df_year[df_year['FL_DATE'] != 'FL_DATE']
        """
        # - Elimina columnes sense nom (Unnamed)
        # - Elimina files que són capçaleres duplicades (on FL_DATE és 'FL_DATE')
        # Això passa perquè alguns CSVs tenen capçaleres repetides dins del fitxer
        # Així evitem errors en la concatenació posterior
        """

        ## Convertir FL_DATE a format datetime
        df_year['FL_DATE'] = pd.to_datetime(df_year['FL_DATE'], errors='coerce')

        ## Llistat de columnes que haurien de ser numèriques
        numeric_cols = [
            'OP_CARRIER_FL_NUM','CRS_DEP_TIME','DEP_TIME','DEP_DELAY','TAXI_OUT',
            'WHEELS_OFF','WHEELS_ON','TAXI_IN','CRS_ARR_TIME','ARR_TIME','ARR_DELAY',
            'CANCELLED','DIVERTED','CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME','AIR_TIME',
            'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY',
            'LATE_AIRCRAFT_DELAY'
        ]

        ## Convertir columnes numèriques a tipus numeric 
        for col in numeric_cols:
            if col in df_year.columns:
                df_year[col] = pd.to_numeric(df_year[col], errors='coerce')
        """
        # - Converteix les columnes listades a tipus numèric
        # - errors='coerce' converteix valors no convertibles a NaN
        # Això ajuda a netejar dades corruptes o mal formatejades
        """

        ## Eliminar files buides o corruptes
        df_year = df_year.dropna(subset=['FL_DATE','OP_CARRIER','ORIGIN','DEST'])
        ## Afegir al llistat de dataframes generats per any
        dfs.append(df_year)

    ## Concatenar i guardar en Parquet per a futures càrregues ràpides
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(PARQUET_PATH, index=False)

print("Dataset correctament carregat.")
print()

print("Dimensions del dataset:", df.shape)
print(df.head())

## Preprocessament i feature engineering per normalitzar les dades
df['MONTH'] = df['FL_DATE'].dt.month  # Extreu el mes (1-12) de la data; captura estacionalitat
df['DAYOFWEEK'] = df['FL_DATE'].dt.dayofweek  # Dia de la setmana (0=Monday ... 6=Sunday); captura patrons setmanals
df['CRSDEPHOUR'] = (df['CRS_DEP_TIME'] // 100).fillna(0).astype(int)  # Converteix CRS_DEP_TIME (HHMM) a hora (HH):
"""
# - fa divisió entera per 100: p.ex. 530 -> 5 (hora)
# - .fillna(0) assigna 0 quan no hi ha hora programada (valor per defecte)
# - .astype(int) garanteix feature entera (0-23)
# - supòsit: CRS_DEP_TIME és numèric en format HHMM; pot ser necessari netejar valors atípics (p.ex. '2400') o strings
"""

## Uniformar noms de columnes
df.columns = df.columns.str.strip().str.upper()
"""
# - Elimina espais en blanc al principi i final dels noms de columnes
# - Converteix noms de columnes a majúscules per uniformitat
# Això ajuda a evitar errors per majúscules/minúscules en accedir a columnes
# i facilita la lectura del codi
# Ex: 'op_carrier' --> 'OP_CARRIER'
"""

## Creem la variable target per a classificació binària: si el vol ha estat cancel·lat o no i la seva correcta classificació
if 'CANCELLED' in df.columns:
    df['TARGET'] = df['CANCELLED'].fillna(0).astype(int) # 1 si cancel·lat, 0 si no
elif 'CANCELLATION_CODE' in df.columns:
    df['TARGET'] = df['CANCELLATION_CODE'].notna().astype(int) # 1 si hi ha codi de cancel·lació, 0 si no
else:
    raise ValueError("No trobo columna de cancel·lació.")
"""
# - Crea la columna 'TARGET' basada en 'CANCELLED' o 'CANCELLATION_CODE'
# - .fillna(0) assegura que valors NaN es considerin com no cancel·lats
# - .notna() converteix valors no nuls a 1 (cancel·lats) i nuls a 0 (no cancel·lats)
"""

print("Percentatge vols cancel·lats:", df['TARGET'].mean())

## Eliminar files sense target definit o que han donat errors
df = df.dropna(subset=['TARGET'])



## Creem un dataset més petit per a proves ràpides (200k files / 61M originals, 0.325%)
features = ['MONTH','DAYOFWEEK','CRSDEPHOUR','OP_CARRIER','ORIGIN','DEST']
from sklearn.model_selection import train_test_split
df_sub, _ = train_test_split(df[features + ['TARGET']], train_size=200000, stratify=df['TARGET'], random_state=14)
df_small = df_sub.reset_index(drop=True)
"""
# - train_test_split s'utilitza aquí per obtenir una mostra més petita, de 200k files
# - stratify=df['TARGET'] assegura que la proporció de vols cancel·lats es manté en la mostra generada
# - reset_index(drop=True) reassigna els índexs del nou DataFrame
"""


## Definim X (predictives) i y (objectiu)
X = df_small[features]
y = df_small['TARGET']

## Definim les columnes numèriques i categòriques per separat
numeric_features = ['MONTH','DAYOFWEEK','CRSDEPHOUR']
cat_features = ['OP_CARRIER','ORIGIN','DEST']

## Preparem els Pipelines de preprocessament per a dades numèriques i categòriques

## Per a dades numèriques, imputem valors faltants amb la mediana
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median'))])
"""
# - SimpleImputer amb 'median' substitueix valors NaN per la mediana de la columna
"""

## Per a dades categòriques, imputem valors faltants amb 'MISSING' i apliquem OneHotEncoding
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
"""
# - SimpleImputer amb 'constant' substitueix valors NaN per 'MISSING'
# - OneHotEncoder converteix categories en variables binàries (0/1)
"""

## ColumnTransformer per aplicar les transformacions adequades en cada tipus de dada
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', cat_transformer, cat_features)
])
"""
# - Aplica numeric_transformer a les columnes numèriques
# - Aplica cat_transformer a les columnes categòriques
"""

## Model (Logistic Regression) dins d'un Pipeline complet
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=300, class_weight='balanced'))
])
"""
# - Preprocessa les dades amb preprocessor abans d'entrenar el model
# - Entrena un model de Logistic Regression amb màxim 300 iteracions
"""

## Entrenament i avaluació del model
## Split train/test del 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=14)
"""
# - train_test_split divideix les dades en conjunt d'entrenament (80%) i de prova (20%)
# - stratify=y assegura que la proporció de classes es manté en ambdós conjunts
# - random_state=14 per a reproduïbilitat en les execucions
"""

## Entrenament
clf.fit(X_train, y_train)

## Avaluació
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
"""
# - predict genera les prediccions de classes (0/1)
# - predict_proba genera les probabilitats per a cada classe; [:,1] agafa la probabilitat de la classe positiva (1)
# El y_proba l'utilitzem per a càlculs com ROC AUC
"""

## Resultats
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
"""
# - classification_report mostra precisió, recall, f1-score per classes
# - roc_auc_score calcula l'AUC-ROC per avaluar la capacitat de classificació
# - confusion_matrix mostra la matriu de confusió per visualitzar prediccions correctes i errònies
"""

